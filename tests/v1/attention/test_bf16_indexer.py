# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer_bf16 import BF16IndexerMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def _builder(dcp=1, pcp=1, non_causal=False):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp
        ),
        model_config=SimpleNamespace(max_model_len=2048),
        attention_config=SimpleNamespace(use_non_causal=non_causal),
    )
    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    return BF16IndexerMetadataBuilder(spec, ["indexer"], config, torch.device("cpu"))


def _common():
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 4, 99], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8, 99], dtype=torch.int32),
        seq_lens=torch.tensor([11, 7, 99], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=11,
        block_table_tensor=torch.tensor([[2], [3], [99]], dtype=torch.int32),
        slot_mapping=torch.tensor([10, 68, 69, 70, -1]),
    )


def test_bf16_metadata_uses_live_device_boundaries_and_replaces_only_cache_mapping():
    """Async speculation may leave CPU lengths ahead of actual device lengths."""
    builder = _builder()
    common = _common()
    metadata = builder.build(0, common)
    assert metadata.query_start_loc.tolist() == [0, 1, 4]
    assert metadata.seq_lens.tolist() == [11, 7]
    assert metadata.block_table.tolist() == [[2], [3]]
    assert metadata.query_start_loc.data_ptr() == common.query_start_loc.data_ptr()
    assert metadata.seq_lens.data_ptr() == common.seq_lens.data_ptr()
    assert metadata.slot_mapping is common.slot_mapping

    new_table = torch.tensor([[4], [5]], dtype=torch.int32)
    new_slots = torch.tensor([20, 78, 79, 80, -1])
    updated = builder.update_block_table(metadata, new_table, new_slots)
    assert updated.block_table is new_table
    assert updated.slot_mapping is new_slots
    assert updated.query_start_loc is metadata.query_start_loc
    assert updated.seq_lens is metadata.seq_lens
    assert metadata.slot_mapping is common.slot_mapping


def test_bf16_capture_covers_context_growth():
    builder = _builder()
    common = _common()
    assert builder.build(0, common).max_seq_len == 11
    assert builder.build_for_cudagraph_capture(common).max_seq_len == 2048


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"dcp": 2}, "context parallelism"),
        ({"pcp": 2}, "context parallelism"),
        ({"non_causal": True}, "causal"),
    ],
)
def test_bf16_indexer_rejects_unsupported_context(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        _builder(**kwargs)
