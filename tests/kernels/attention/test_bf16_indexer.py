# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.v1.attention.ops.bf16_indexer import (
    bf16_indexer,
    bf16_indexer_logits,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="BF16 DSA indexer requires CUDA"
)


def _inputs(num_heads: int, head_dim: int):
    torch.manual_seed(42)
    device = "cuda"
    block_size = 64
    num_tokens = 10
    q = torch.randn(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(num_tokens, num_heads, device=device)
    weights *= (num_heads * head_dim) ** -0.5
    # The allocator can pad between pages. Neither reads nor writes may use
    # block_size * head_dim in place of the actual page stride.
    storage = torch.randn(
        8, block_size * head_dim + 32, dtype=torch.bfloat16, device=device
    )
    cache = storage.as_strided(
        (8, block_size, head_dim), (storage.stride(0), head_dim, 1)
    )
    block_table = torch.tensor(
        [[4, 1], [6, 0], [3, 0], [0, 0]], dtype=torch.int32, device=device
    )
    # Multi-token decode, single-token decode, chunked prefill, padding.
    query_start_loc = torch.tensor([0, 2, 3, 8, 8], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([67, 19, 8, 0], dtype=torch.int32, device=device)
    slots = [65, 66, 6 * 64 + 18, *range(3 * 64 + 3, 3 * 64 + 8), -1]
    return dict(
        q=q,
        k=k,
        kv_cache=cache,
        weights=weights,
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=80,
    ), storage


def _reference_logits(inputs):
    q = inputs["q"].float().cpu()
    cache = inputs["kv_cache"].float().cpu()
    weights = inputs["weights"].cpu()
    table = inputs["block_table"].cpu()
    starts = inputs["query_start_loc"].tolist()
    lengths = inputs["seq_lens"].tolist()
    scores = torch.full((q.shape[0], inputs["max_seq_len"]), -float("inf"))
    block_size = cache.shape[1]
    for req, length in enumerate(lengths):
        for token in range(starts[req], starts[req + 1]):
            causal_length = length - (starts[req + 1] - token) + 1
            pos = torch.arange(causal_length)
            keys = cache[table[req, pos // block_size].long(), pos % block_size]
            dot = q[token] @ keys.T
            scores[token, :causal_length] = (
                dot.clamp_min(0) * weights[token, :, None]
            ).sum(0)
    return scores


@pytest.mark.parametrize("num_heads,head_dim", [(5, 64), (64, 128)])
def test_bf16_paged_logits_preserve_causality_and_fp32_scores(num_heads, head_dim):
    inputs, _ = _inputs(num_heads, head_dim)
    reference = _reference_logits(inputs)
    kwargs = {
        key: value for key, value in inputs.items() if key not in ("k", "slot_mapping")
    }
    actual = bf16_indexer_logits(**kwargs).cpu()
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-4)
    # Chunk offsets must index the original query and request metadata.
    chunk = bf16_indexer_logits(**kwargs, query_start=1, num_query_tokens=4)
    torch.testing.assert_close(chunk.cpu(), reference[1:5], rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("topk,chunk_and_capture", [(4, False), (128, True)])
def test_bf16_indexer_updates_pages_and_selects_only_valid_tokens(
    monkeypatch: pytest.MonkeyPatch, topk: int, chunk_and_capture: bool
):
    inputs, storage = _inputs(32, 128)
    expected_storage = storage.clone()
    expected_cache = expected_storage.as_strided(
        inputs["kv_cache"].shape, inputs["kv_cache"].stride()
    )
    for token, slot in enumerate(inputs["slot_mapping"].tolist()):
        if slot >= 0:
            expected_cache[slot // 64, slot % 64] = inputs["k"][token]
    if chunk_and_capture:
        monkeypatch.setattr(envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 0)
    output = torch.full(
        (inputs["q"].shape[0], topk), 123, device="cuda", dtype=torch.int32
    )

    def run():
        return bf16_indexer(**inputs, topk_tokens=topk, topk_indices=output)

    assert run() is output
    if chunk_and_capture:
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        output.fill_(123)
        graph.replay()

    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)
    reference = _reference_logits(inputs)
    for actual, scores in zip(output.cpu(), reference):
        valid_count = int(torch.isfinite(scores).sum())
        count = min(topk, valid_count)
        selected = actual[actual >= 0].long()
        assert selected.numel() == count
        assert selected.unique().numel() == count
        assert int((actual == -1).sum()) == topk - count
        if count:
            assert torch.all(selected < valid_count)
            # Compare scores rather than indices because ties may be ordered
            # differently by the reference and device top-k implementations.
            torch.testing.assert_close(
                scores[selected].sort().values,
                scores.topk(count).values.sort().values,
                rtol=2e-4,
                atol=2e-4,
            )
