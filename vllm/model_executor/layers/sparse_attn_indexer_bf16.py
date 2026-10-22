# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer_bf16 import BF16IndexerMetadata
from vllm.v1.attention.ops.bf16_indexer import bf16_indexer


@eager_break_during_capture
def sparse_attn_indexer_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    layer_name: LayerNameType,
    topk_tokens: int,
    max_model_len: int,
) -> None:
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        # Account for the bounded score workspace and top-k temporaries in the
        # memory profile, before the cache is allocated.
        budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        rows = min(q.shape[0], max(1, budget // (4 * max_model_len)))
        size = rows * (4 * max_model_len + 16 * min(topk_tokens, max_model_len))
        workspace = torch.empty(size, dtype=torch.uint8, device=q.device)
        topk_indices[: q.shape[0]].fill_(-1)
        del workspace
        return

    layer_metadata = metadata[_resolve_layer_name(layer_name)]
    assert isinstance(layer_metadata, BF16IndexerMetadata)
    bf16_indexer(
        q,
        k,
        kv_cache,
        weights,
        layer_metadata.slot_mapping,
        layer_metadata.block_table,
        layer_metadata.query_start_loc,
        layer_metadata.seq_lens,
        layer_metadata.max_seq_len,
        topk_tokens,
        topk_indices,
    )


def sparse_attn_indexer_bf16_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    layer_name: LayerNameType,
    topk_tokens: int,
    max_model_len: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="sparse_attn_indexer_bf16",
    op_func=sparse_attn_indexer_bf16,
    mutates_args=["kv_cache", "topk_indices"],
    fake_impl=sparse_attn_indexer_bf16_fake,
)


class BF16SparseAttnIndexer(nn.Module):
    def __init__(
        self,
        k_cache,
        topk_tokens: int,
        topk_indices_buffer: torch.Tensor,
        max_model_len: int,
    ) -> None:
        super().__init__()
        self.k_cache = k_cache
        self.topk_tokens = topk_tokens
        self.topk_indices_buffer = topk_indices_buffer
        self.max_model_len = max_model_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.vllm.sparse_attn_indexer_bf16(
            q,
            k,
            weights,
            self.k_cache.kv_cache,
            self.topk_indices_buffer,
            _encode_layer_name(self.k_cache.prefix),
            self.topk_tokens,
            self.max_model_len,
        )
        return self.topk_indices_buffer
