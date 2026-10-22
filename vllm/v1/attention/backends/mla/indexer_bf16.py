# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass
class BF16IndexerMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_seq_len: int


class BF16IndexerBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER_BF16"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("The BF16 indexer is a cache-only attention layer.")

    @staticmethod
    def get_builder_cls() -> type["BF16IndexerMetadataBuilder"]:
        return BF16IndexerMetadataBuilder


class BF16IndexerMetadataBuilder(AttentionMetadataBuilder[BF16IndexerMetadata]):
    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = True
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        parallel = vllm_config.parallel_config
        if (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise ValueError(
                "The BF16 DSA indexer does not support context parallelism."
            )
        if vllm_config.attention_config.use_non_causal:
            raise ValueError("The BF16 DSA indexer only supports causal attention.")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BF16IndexerMetadata:
        common = common_attn_metadata
        return BF16IndexerMetadata(
            query_start_loc=common.query_start_loc[: common.num_reqs + 1],
            seq_lens=common.seq_lens[: common.num_reqs],
            block_table=common.block_table_tensor[: common.num_reqs],
            slot_mapping=common.slot_mapping,
            max_seq_len=common.max_seq_len,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> BF16IndexerMetadata:
        metadata = self.build(0, common_attn_metadata)
        # Captured launch sizes must cover later replays with longer contexts.
        metadata.max_seq_len = self.vllm_config.model_config.max_model_len
        return metadata

    def update_block_table(
        self,
        metadata: BF16IndexerMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> BF16IndexerMetadata:
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping)
