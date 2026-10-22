# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unquantized paged DSA indexer with BF16 inputs and FP32 scores."""

import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton


@triton.jit
def _bf16_indexer_cache_kernel(
    k_ptr,
    cache_ptr,
    slots_ptr,
    k_stride_0,
    k_stride_1,
    cache_stride_0,
    cache_stride_1,
    cache_stride_2,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots_ptr + token).to(tl.int64)
    if slot >= 0:
        dim = tl.arange(0, BLOCK_D)
        k = tl.load(
            k_ptr + token.to(tl.int64) * k_stride_0 + dim * k_stride_1,
            mask=dim < HEAD_DIM,
            other=0,
        )
        cache_offset = (
            (slot // PAGE_SIZE) * cache_stride_0
            + (slot % PAGE_SIZE) * cache_stride_1
            + dim * cache_stride_2
        )
        tl.store(cache_ptr + cache_offset, k, mask=dim < HEAD_DIM)


@triton.jit(do_not_specialize=["query_start", "num_reqs", "max_seq_len"])
def _bf16_indexer_logits_kernel(
    q_ptr,
    cache_ptr,
    weights_ptr,
    block_table_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    logits_ptr,
    query_start,
    num_reqs,
    max_seq_len,
    q_stride_0,
    q_stride_1,
    q_stride_2,
    cache_stride_0,
    cache_stride_1,
    cache_stride_2,
    weights_stride_0,
    weights_stride_1,
    block_table_stride,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # A one-dimensional grid also supports contexts larger than 65535 tiles.
    num_key_tiles = tl.cdiv(max_seq_len, BLOCK_N)
    row = tl.program_id(0) // num_key_tiles
    key_tile = tl.program_id(0) % num_key_tiles
    token = row + query_start
    keys = key_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    scores = tl.full((BLOCK_N,), -float("inf"), tl.float32)
    num_tokens = tl.load(query_start_loc_ptr + num_reqs)

    if token < num_tokens:
        # upper_bound handles zero-length requests and graph padding as well.
        lo = 0
        hi = num_reqs
        while lo < hi:
            mid = (lo + hi) // 2
            end = tl.load(query_start_loc_ptr + mid + 1)
            before_end = token < end
            hi = tl.where(before_end, mid, hi)
            lo = tl.where(before_end, lo, mid + 1)
        req = lo
        start = tl.load(query_start_loc_ptr + req)
        end = tl.load(query_start_loc_ptr + req + 1)
        seq_len = tl.load(seq_lens_ptr + req)
        causal_end = tl.maximum(seq_len - (end - start) + token - start + 1, 0)
        causal_end = tl.minimum(causal_end, seq_len)
        valid_keys = (keys < causal_end) & (keys < max_seq_len)

        if key_tile * BLOCK_N < causal_end:
            heads = tl.arange(0, BLOCK_H)
            dims = tl.arange(0, BLOCK_D)
            q = tl.load(
                q_ptr
                + token.to(tl.int64) * q_stride_0
                + heads[:, None] * q_stride_1
                + dims[None, :] * q_stride_2,
                mask=(heads[:, None] < NUM_HEADS) & (dims[None, :] < HEAD_DIM),
                other=0,
            )
            blocks = tl.load(
                block_table_ptr
                + tl.cast(req, tl.int64) * block_table_stride
                + keys // PAGE_SIZE,
                mask=valid_keys,
                other=0,
            ).to(tl.int64)
            k = tl.load(
                cache_ptr
                + blocks[None, :] * cache_stride_0
                + (keys[None, :] % PAGE_SIZE) * cache_stride_1
                + dims[:, None] * cache_stride_2,
                mask=valid_keys[None, :] & (dims[:, None] < HEAD_DIM),
                other=0,
            )
            dots = tl.dot(q, k, out_dtype=tl.float32)
            weights = tl.load(
                weights_ptr
                + token.to(tl.int64) * weights_stride_0
                + heads * weights_stride_1,
                mask=heads < NUM_HEADS,
                other=0,
            ).to(tl.float32)
            scores = tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0)
            scores = tl.where(valid_keys, scores, -float("inf"))

    tl.store(
        logits_ptr + row.to(tl.int64) * max_seq_len + keys,
        scores,
        mask=keys < max_seq_len,
    )


def bf16_indexer_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Insert unquantized keys, ignoring negative and padded token slots."""
    assert k.dtype == kv_cache.dtype == torch.bfloat16
    num_tokens = min(k.shape[0], slot_mapping.numel())
    if num_tokens == 0:
        return
    _bf16_indexer_cache_kernel[(num_tokens,)](
        k,
        kv_cache,
        slot_mapping,
        *k.stride(),
        *kv_cache.stride(),
        HEAD_DIM=k.shape[-1],
        PAGE_SIZE=kv_cache.shape[1],
        BLOCK_D=triton.next_power_of_2(k.shape[-1]),
    )


def bf16_indexer_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    query_start: int = 0,
    num_query_tokens: int | None = None,
) -> torch.Tensor:
    """Compute causal FP32 index scores for a slice of the token batch.

    The score for a key is the weighted sum over heads of the rectified
    BF16 query/key dot products. Weights include both model scale factors.
    Columns are request-local key positions; invalid columns contain -inf.
    """
    assert q.dtype == kv_cache.dtype == torch.bfloat16
    if num_query_tokens is None:
        num_query_tokens = q.shape[0] - query_start
    logits = torch.empty(
        (num_query_tokens, max_seq_len), device=q.device, dtype=torch.float32
    )
    if num_query_tokens == 0 or max_seq_len == 0:
        return logits
    if seq_lens.numel() == 0:
        return logits.fill_(-float("inf"))
    block_n = 64
    _bf16_indexer_logits_kernel[
        (num_query_tokens * triton.cdiv(max_seq_len, block_n),)
    ](
        q,
        kv_cache,
        weights,
        block_table,
        query_start_loc,
        seq_lens,
        logits,
        query_start,
        seq_lens.numel(),
        max_seq_len,
        *q.stride(),
        *kv_cache.stride(),
        *weights.stride(),
        block_table.stride(0),
        HEAD_DIM=q.shape[-1],
        NUM_HEADS=q.shape[1],
        PAGE_SIZE=kv_cache.shape[1],
        BLOCK_H=max(16, triton.next_power_of_2(q.shape[1])),
        BLOCK_D=max(16, triton.next_power_of_2(q.shape[-1])),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return logits


def bf16_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    """Update the BF16 paged cache and select causal DSA token indices.

    Logits are chunked along the query dimension to bound temporary memory.
    Graph-padded queries and unfilled top-k entries receive the -1 sentinel.
    """
    assert topk_indices.dtype == torch.int32
    bf16_indexer_cache(k, kv_cache, slot_mapping)
    topk_indices.fill_(-1)
    selected = min(topk_tokens, max_seq_len)
    if selected == 0 or q.shape[0] == 0:
        return topk_indices
    max_logits_bytes = max(1, envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024)
    rows_per_chunk = max(1, max_logits_bytes // (max_seq_len * 4))
    for start in range(0, q.shape[0], rows_per_chunk):
        rows = min(rows_per_chunk, q.shape[0] - start)
        logits = bf16_indexer_logits(
            q,
            kv_cache,
            weights,
            block_table,
            query_start_loc,
            seq_lens,
            max_seq_len,
            start,
            rows,
        )
        values, indices = torch.topk(logits, selected, dim=-1, sorted=False)
        indices.masked_fill_(values == -float("inf"), -1)
        topk_indices[start : start + rows, :selected].copy_(indices)
        del logits, values, indices
    return topk_indices
