# SPDX-License-Identifier: Apache-2.0
"""Compressed index slots must retain the ownership of their original KV page."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.kv_shard_hooks import get_kv_shard_pool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.page_interleave import PageInterleavePlacement, PageShardSpec
from sglang.srt.mem_cache.page_interleave_pool import (
    PageInterleaveDSATokenToKVPool,
    _PageInterleaveIndexKeyCache,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_index_cache(kpool, rank=1):
    # CPU storage exercises the production address translation without creating
    # a CUDA communicator. One physical KV page contains 64 compressed slots.
    pool = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
    pool.page_size = 64 * kpool
    pool.index_kpool = kpool
    pool.index_page_size = 64
    pool.index_head_dim = 128
    pool.quant_block_size = 128
    pool.start_layer = 0
    pool.shard_rank = rank
    pool.shard_spec = PageShardSpec(
        rank, 2, pool.page_size, 4 * pool.page_size, 2 * pool.page_size
    )
    pool.placement = PageInterleavePlacement(pool.shard_spec)
    pool._shard_extend_active = True
    pool._epoch = 1
    pool._index_write_plan = None
    pool._page_pos = torch.tensor([6, 6, 3, 1, 5, 2], dtype=torch.int32)
    pool._slots = [
        SimpleNamespace(tensors={"index_k": torch.zeros((7, 8448), dtype=torch.uint8)})
    ]
    cache = _PageInterleaveIndexKeyCache.__new__(_PageInterleaveIndexKeyCache)
    cache.pool = pool
    cache.buffer = [torch.zeros((4, 8448), dtype=torch.uint8)]
    return cache


class TestPageShardedKPool(CustomTestCase):
    def test_glm_rejects_plain_tp_sharding_before_pool_construction(self):
        from sglang.srt.arg_groups import kv_shard_hook

        cfg = ServerArgs(
            model_path="dummy",
            enable_kv_cache_sharding=True,
            disaggregation_transfer_backend="mooncake",
            pp_size=1,
            attn_cp_size=1,
            enable_prefill_cp=False,
            enable_cp_tp_group_sharing=False,
            cp_strategy="interleave",
        )
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"], index_topk=2048
            )
        )
        # Plain TP otherwise chooses the TP group in pool sizing but the
        # trivial CP group in the hybrid PD manager, losing foreign DSA pages.
        with (
            patch.object(kv_shard_hook, "model_config_of", return_value=model),
            self.assertRaisesRegex(ValueError, "requires prefill CP"),
        ):
            kv_shard_hook.handle_kv_cache_sharding(cfg, gpu_mem=80)

    def test_hicache_pairs_host_rows_with_owned_local_device_rows(self):
        from sglang.srt.layers.attention.kv_shard_hooks import (
            filter_kv_shard_host_indices,
        )

        pool = make_index_cache(4).pool
        device = torch.cat((torch.arange(512, 768), torch.arange(1280, 1536)))
        host = torch.arange(768, 1280)
        host_rows, device_rows = filter_kv_shard_host_indices(pool, host, device)
        self.assertEqual(host_rows.tolist(), list(range(1024, 1280)))
        self.assertEqual(device_rows.tolist(), list(range(512, 768)))

    def test_hicache_restore_keeps_owner_phase_across_fragments(self):
        from sglang.srt.mem_cache.allocator.page_interleave import (
            PageInterleavePoolAllocator,
        )

        allocator = PageInterleavePoolAllocator(
            size=2048,
            physical_page_size=256,
            shard_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        spans = [(768, 2), (512, 1)]
        rows = allocator.alloc_restore_spans(spans)
        self.assertEqual((rows[::256] // 256 % 4).tolist(), [2, 3, 0, 1, 2])
        allocator.free(rows)
        restored = allocator.alloc_restore_spans(spans)
        self.assertEqual((restored[::256] // 256 % 4).tolist(), [2, 3, 0, 1, 2])
        self.assertEqual(len(set(restored.tolist())), 1280)

    def test_draft_budget_covers_all_target_owner_stripes(self):
        from sglang.srt.model_executor.pool_configurator import _dflash_draft_cell_size
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        kvc = SimpleNamespace(
            is_draft_worker=False,
            kv_shard_size=4,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_aux_config=SimpleNamespace(dflash_draft_cell_size_per_token=10240),
        )
        with get_parallel().override(attn_dcp_size=1):
            self.assertEqual(_dflash_draft_cell_size(kvc), 40960)
            kvc.is_draft_worker = True
            self.assertEqual(_dflash_draft_cell_size(kvc), 0)

    def test_draft_address_space_includes_every_reserved_owner_page(self):
        from sglang.srt.mem_cache.allocator.page_interleave import (
            PageInterleavePoolAllocator,
        )

        allocator = PageInterleavePoolAllocator(
            size=1024,
            physical_page_size=256,
            shard_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        highest = (
            max(int(pages.max()) for pages in allocator.class_free_pages) * 256 + 255
        )
        # Ordinary draft pools add one padding page to their advertised size.
        self.assertLess(highest, allocator.draft_virtual_id_space + 256)
        self.assertEqual(allocator.draft_virtual_id_space, 4864)

    def test_compressed_slots_follow_latent_page_owner(self):
        # Logical pages 2,3,4,5 with compressed offsets 0,63,1,62.
        # Rank 1 owns pages 3 and 5: local pages 1 and 2 respectively.
        for kpool in (1, 2, 4):
            with self.subTest(kpool=kpool):
                cache = make_index_cache(kpool)
                targets = cache.write_targets(0, torch.tensor([128, 255, 257, 382]))
                self.assertEqual(len(targets), 2)
                scratch, local = targets
                self.assertEqual(scratch[1].tolist(), [192, 127, 321, 190])
                self.assertIsNone(scratch[2])
                self.assertEqual(local[1].tolist(), [127, 190])
                self.assertEqual(local[2].tolist(), [1, 3])

    def test_immutable_batch_plan_reuses_mapping_across_layers(self):
        cache = make_index_cache(4)
        cache.buffer.append(torch.zeros_like(cache.buffer[0]))
        cache.pool._slots.append(
            SimpleNamespace(
                tensors={"index_k": torch.zeros((7, 8448), dtype=torch.uint8)}
            )
        )
        loc = torch.tensor([128, 255, 257, 382])
        plan = object()
        # Counting nonzero verifies that the device-to-host synchronization
        # occurs once for this immutable batch plan, rather than once per layer.
        with patch.object(torch, "nonzero", wraps=torch.nonzero) as nonzero:
            first = cache.write_targets(0, loc, immutable_plan=plan)
            second = cache.write_targets(1, loc, immutable_plan=plan)
            self.assertEqual(nonzero.call_count, 1)
        self.assertIs(first[1][1], second[1][1])
        self.assertIs(first[1][2], second[1][2])
        self.assertIs(second[1][0], cache.buffer[1])
        self.assertIs(second[0][0], cache.pool._slots[1].tensors["index_k"])
        self.assertEqual(second[1][1].tolist(), [127, 190])

    def test_immutable_plan_cache_invalidates_for_new_batch_or_plan(self):
        cache = make_index_cache(4)
        loc = torch.tensor([128, 255])
        plan = object()
        first = cache.write_targets(0, loc, immutable_plan=plan)
        self.assertEqual(first[1][1].tolist(), [127])
        # Even when a caller retains its plan object, the next batch must
        # rebuild both the ownership and scratch mapping.
        cache.pool._epoch += 1
        loc.copy_(torch.tensor([319, 320]))
        second = cache.write_targets(0, loc, immutable_plan=plan)
        self.assertEqual(second[1][1].tolist(), [128])
        loc.copy_(torch.tensor([128, 255]))
        third = cache.write_targets(0, loc, immutable_plan=object())
        self.assertEqual(third[1][1].tolist(), [127])

    def test_reused_location_tensor_recomputes_owner(self):
        cache = make_index_cache(4)
        loc = torch.tensor([128, 255])
        self.assertEqual(cache.write_targets(0, loc)[1][2].tolist(), [1])
        loc.copy_(torch.tensor([319, 320]))
        # Page 4 is foreign; page 5 offset 0 belongs to this rank.
        targets = cache.write_targets(0, loc)
        self.assertEqual(targets[1][1].tolist(), [128])
        self.assertEqual(targets[1][2].tolist(), [1])

    def test_hybrid_discovery_returns_only_full_attention_pool(self):
        hybrid = HybridLinearKVPool.__new__(HybridLinearKVPool)
        hybrid.full_kv_pool = make_index_cache(4).pool
        hybrid.mamba_pool = object()
        self.assertIs(get_kv_shard_pool(hybrid), hybrid.full_kv_pool)
        self.assertIsNone(get_kv_shard_pool(hybrid.mamba_pool))

    def test_hybrid_index_writer_maps_model_layer_to_dense_child(self):
        from sglang.srt.layers.attention.dsa.dsa_indexer import _index_k_write_targets

        cache = make_index_cache(4)
        cache.pool.index_key_cache = cache
        hybrid = HybridLinearKVPool.__new__(HybridLinearKVPool)
        hybrid.full_kv_pool = cache.pool
        hybrid.full_attention_layer_id_mapping = {43: 0}
        targets = _index_k_write_targets(hybrid, 43, torch.tensor([255]))
        self.assertEqual(targets[1][1].tolist(), [127])

    def test_scratch_budget_counts_pooled_index_keys_and_latent_scales(self):
        from sglang.srt.mem_cache.page_interleave import _page_shard_row_bytes
        from sglang.srt.runtime_context import get_disagg, get_exec, publish
        from sglang.srt.server_args import ServerArgs

        publish(ServerArgs(model_path="dummy"), role="test")

        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"],
                index_topk=2048,
                index_head_dim=128,
                index_kpool=4,
            ),
            kv_lora_rank=512,
            qk_rope_head_dim=0,
        )
        kvc = SimpleNamespace(
            model_config=model, use_mla_backend=True, kv_cache_dtype=torch.float8_e4m3fn
        )
        with get_disagg().override(disaggregation_mode="prefill"):
            for backend, expected in (("trtllm", 545), ("flashmla_sparse", 561)):
                with (
                    self.subTest(backend=backend),
                    get_exec().kernel.override(dsa_prefill_backend=backend),
                ):
                    # Indexer: (128 FP8 bytes + 4 scale bytes) / four tokens.
                    # Packed MLA additionally carries four FP32 latent scales.
                    self.assertEqual(_page_shard_row_bytes(kvc), expected)


if __name__ == "__main__":
    unittest.main()
