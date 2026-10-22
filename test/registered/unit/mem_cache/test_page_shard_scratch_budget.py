# SPDX-License-Identifier: Apache-2.0
"""Operator-sized gather scratch must retain full-context request admission."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.page_interleave import (
    compute_page_shard_scratch_bytes,
    make_page_shard_spec,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPageShardScratchBudget(CustomTestCase):
    def setUp(self):
        self.kvc = SimpleNamespace(
            page_size=256,
            use_mla_backend=True,
            kv_cache_dtype=torch.bfloat16,
            model_config=SimpleNamespace(
                context_len=1048577,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=SimpleNamespace(architectures=["DeepseekV2ForCausalLM"]),
            ),
        )
        topology = patch(
            "sglang.srt.mem_cache.page_interleave.get_kv_shard_group_info",
            return_value=(0, 2),
        )
        topology.start()
        self.addCleanup(topology.stop)

    def test_default_preserves_eight_context_budget(self):
        with get_context().override_server_args(chunked_prefill_size=8192):
            spec = make_page_shard_spec(self.kvc)
        self.assertEqual(spec.max_prefix_tokens, 8 * 1049088)
        self.assertEqual(spec.chunk_tokens, 8192)

    def test_two_context_budget_charges_both_buffers_without_shortening_context(self):
        with get_context().override_server_args(
            chunked_prefill_size=8192, kv_cache_sharding_prefix_contexts=2
        ):
            spec = make_page_shard_spec(self.kvc)
            actual_bytes = compute_page_shard_scratch_bytes(self.kvc)
        self.assertEqual(spec.max_prefix_tokens, 2 * 1049088)
        self.assertEqual(actual_bytes, 2 * (2 * 1049088 + 8192 + 256) * 576 * 2)
        self.assertEqual(self.kvc.model_config.context_len, 1048577)

    def test_excess_prefix_defers_without_consuming_scratch_budget(self):
        with get_context().override_server_args(
            chunked_prefill_size=8192, kv_cache_sharding_prefix_contexts=2
        ):
            spec = make_page_shard_spec(self.kvc)
        adder = SimpleNamespace(
            kv_shard_scratch_spec=spec,
            kv_shard_granule=256,
            kv_shard_size=2,
            kv_shard_block_bound_pages=0,
            kv_shard_chunk_pages=0,
        )
        reserve = PrefillAdder._kv_shard_reserve_scratch
        self.assertTrue(reserve(adder, 1048576, 256))
        self.assertTrue(reserve(adder, 1048576, 256))
        previous = (adder.kv_shard_block_bound_pages, adder.kv_shard_chunk_pages)
        self.assertFalse(reserve(adder, 1048576, 256))
        self.assertEqual(
            (adder.kv_shard_block_bound_pages, adder.kv_shard_chunk_pages), previous
        )

    def test_nonpositive_prefix_context_budget_is_rejected(self):
        for contexts in (0, -1):
            with (
                self.subTest(contexts=contexts),
                get_context().override_server_args(
                    chunked_prefill_size=8192,
                    kv_cache_sharding_prefix_contexts=contexts,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "prefix-contexts.*positive"):
                    make_page_shard_spec(self.kvc)


if __name__ == "__main__":
    unittest.main()
