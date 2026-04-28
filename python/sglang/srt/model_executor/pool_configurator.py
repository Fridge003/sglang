"""Memory pool configurator for DSv4 compressed-attention models.

Resolves full / swa / c4 / c128 + c4_state / c128_state pool sizes from
available GPU memory. Used by ``ModelRunnerKVCacheMixin._resolve_memory_pool_config``
when the model is DSv4 compressed and hybrid SWA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseekv4_memory_pool import get_compress_state_ring_size
from sglang.srt.utils.common import get_available_gpu_memory

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import MemoryPoolConfig

logger = logging.getLogger(__name__)


@dataclass
class DSv4PoolSizes:
    full_max_total_num_tokens: int
    swa_max_total_num_tokens: int
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int


class DSv4PoolConfigurator:
    """Resolves DSv4 compressed-attention pool sizes into a ``MemoryPoolConfig``.

    Replaces the legacy ``DSv4MemoryCalculator`` (memory_profiler.py) +
    ``set_num_tokens_hybrid_swa_compress`` (mixin method) split: one class
    that owns both the byte-budget math and the resolve-into-config flow.
    """

    def __init__(self, mr: ModelRunner):
        self._mr = mr
        cfg = mr.model_config
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.indexer_head_dim = cfg.index_head_dim
        self.compression_ratios = cfg.compress_ratios
        self.swa_page_size = cfg.window_size
        self.page_size = mr.server_args.page_size
        self.swa_ratio = mr.server_args.swa_full_tokens_ratio
        self.is_speculative = mr.server_args.speculative_algorithm is not None
        self.c4_shrink_factor = (
            envs.SGLANG_OPT_HISPARSE_C4_SHRINK.get() if mr.enable_hisparse else 1
        )
        assert self.c4_shrink_factor >= 1
        if self.c4_shrink_factor > 1:
            logger.info(
                f"HiSparse c4 pool shrink factor = {self.c4_shrink_factor} "
                f"(set via SGLANG_OPT_HISPARSE_C4_SHRINK)"
            )

        assert (
            self.page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"

        self.c4_ring_size = get_compress_state_ring_size(4, self.is_speculative)
        self.c128_ring_size = get_compress_state_ring_size(128, self.is_speculative)

        self.num_layers_total = len(self.compression_ratios)
        self.num_layers_ca4 = sum(1 for r in self.compression_ratios if r == 4)
        self.num_layers_ca128 = sum(1 for r in self.compression_ratios if r == 128)

        self.bytes_per_full_token = self._get_bytes_per_full_token()

    def _get_bytes_per_full_token(self) -> float:
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8

        quant_block_size = 128
        indexer_bytes = (
            self.indexer_head_dim + self.indexer_head_dim // quant_block_size * 4
        )

        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        state_dtype_size = 4
        c4_state_bytes = 2 * 2 * attn_head_dim * state_dtype_size
        # Online c128 stores (max, sum, kv) per slot (3*head_dim) instead of
        # raw (kv, score) (2*head_dim). Combined with ring_size=1 this still
        # nets a large reduction (~3/256x) but the per-slot bytes go up.
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        c128_state_bytes = (
            (3 if c128_online else 2 * 1) * attn_head_dim * state_dtype_size
        )
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * state_dtype_size

        c4_state_ratio = self.c4_ring_size / self.swa_page_size
        c128_state_ratio = self.c128_ring_size / self.swa_page_size

        c4_frac = 1 / (4 * self.c4_shrink_factor)
        return (
            self.swa_ratio * kv_bytes * self.num_layers_total
            + c4_frac * kv_bytes * self.num_layers_ca4
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
            + self.swa_ratio * c4_state_ratio * c4_state_bytes * self.num_layers_ca4
            + self.swa_ratio
            * c128_state_ratio
            * c128_state_bytes
            * self.num_layers_ca128
            + self.swa_ratio
            * c4_state_ratio
            * c4_indexer_state_bytes
            * self.num_layers_ca4
        )

    def _calculate_pool_sizes(self, available_bytes: int) -> DSv4PoolSizes:
        full_token = int(available_bytes / self.bytes_per_full_token)
        full_token = full_token // self.page_size * self.page_size
        swa_tokens = int(full_token * self.swa_ratio) // self.page_size * self.page_size

        pool_sizes = DSv4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            c128_max_total_num_tokens=full_token // 128,
            c4_state_pool_size=swa_tokens // self.swa_page_size * self.c4_ring_size,
            c128_state_pool_size=swa_tokens // self.swa_page_size * self.c128_ring_size,
        )
        logger.info(
            f"DSv4 memory calculation: "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"full_token={full_token}"
        )
        return pool_sizes

    def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
        post_model_load_memory = get_available_gpu_memory(
            self._mr.device,
            self._mr.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self._mr.mem_fraction_static
        )
        available_bytes = int(rest_memory * (1 << 30))
        logger.info(
            f"DSv4 memory profiling: post_model_load_memory={post_model_load_memory:.2f} GB, "
            f"pre_model_load_memory={pre_model_load_memory:.2f} GB, "
            f"mem_fraction_static={self._mr.mem_fraction_static:.2f}, "
            f"rest_memory={rest_memory:.2f} GB"
        )
        return available_bytes

    def resolve(self, pre_model_load_memory: int) -> MemoryPoolConfig:
        """Profile + size + apply user cap, returning a complete MemoryPoolConfig."""
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            MemoryPoolConfig,
        )

        # Compressed attention always stores c4/c128 states in fp32. _init_pools
        # reads self.state_dtype when constructing DeepSeekV4TokenToKVPool.
        self._mr.state_dtype = torch.float32
        logger.info(
            f"DSv4 compressed attention: kv_cache_dtype={self._mr.kv_cache_dtype}"
        )
        logger.info(f"DSv4 compressed attention: state_dtype={self._mr.state_dtype}")

        # Online c128 keeps a single in-progress (max, sum, kv) state per index
        # and assumes a strict forward-only schedule. Speculative decode (MTP)
        # would need rollback / replay across draft and verify, which the
        # online path doesn't support yet.
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            assert (
                self._mr.spec_algorithm.is_none()
            ), "SGLANG_OPT_USE_ONLINE_COMPRESS does not support speculative decode (MTP) yet"
            logger.info("DSv4 compressed attention: online c128 enabled (ring_size=1)")

        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        if self.is_speculative:
            # Reserve memory for the speculative draft worker's layers.
            draft_layers = 1
            target_layers = self.num_layers_total
            available_bytes = int(
                available_bytes * target_layers / (target_layers + draft_layers)
            )

        pool_sizes = self._calculate_pool_sizes(available_bytes)

        # Apply user max_total_tokens cap by recomputing from cap.
        user_cap = self._mr.server_args.max_total_tokens
        if user_cap is not None and pool_sizes.full_max_total_num_tokens > user_cap:
            pool_sizes = self._calculate_pool_sizes(
                int(user_cap * self.bytes_per_full_token)
            )

        full = pool_sizes.full_max_total_num_tokens
        swa = pool_sizes.swa_max_total_num_tokens
        logger.info(
            f"DSv4 pool sizes: full={full}, swa={swa}, "
            f"c4={pool_sizes.c4_max_total_num_tokens}, "
            f"c128={pool_sizes.c128_max_total_num_tokens}, "
            f"c4_state={pool_sizes.c4_state_pool_size}, "
            f"c128_state={pool_sizes.c128_state_pool_size}"
        )

        return MemoryPoolConfig(
            max_total_num_tokens=full,
            max_running_requests=self._mr._resolve_max_num_reqs(full),
            full_max_total_num_tokens=full,
            swa_max_total_num_tokens=swa,
            c4_max_total_num_tokens=pool_sizes.c4_max_total_num_tokens,
            c128_max_total_num_tokens=pool_sizes.c128_max_total_num_tokens,
            c4_state_pool_size=pool_sizes.c4_state_pool_size,
            c128_state_pool_size=pool_sizes.c128_state_pool_size,
            mem_fraction_static=self._mr.server_args.mem_fraction_static,
        )
