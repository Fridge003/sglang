"""Unit tests for srt/layers/moe/topk.py."""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import topk as moe_topk

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class TestBiasedGroupedTopkGpuFallback(CustomTestCase):
    def test_grouped_topk_fast_path_appends_shared_expert(self):
        gating_output = torch.tensor(
            [[0.1, 0.4, 0.3, 0.2], [0.6, 0.1, 0.2, 0.1]], dtype=torch.float32
        )
        correction_bias = torch.zeros_like(gating_output)
        fake_weights = torch.tensor([[0.7, 0.2], [0.6, 0.3]], dtype=torch.float32)
        fake_ids = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        with (
            patch.object(moe_topk, "_is_cuda", True),
            patch.object(moe_topk, "_is_musa", False),
            patch.object(moe_topk, "_use_aiter", False),
            patch.object(moe_topk, "fused_topk_deepseek", None),
            patch.object(moe_topk, "moe_fused_gate", None),
            patch.object(moe_topk, "is_power_of_two", return_value=False),
            patch(
                "sglang.jit_kernel.grouped_topk.grouped_topk",
                return_value=(fake_weights, fake_ids),
            ) as mock_grouped_topk,
        ):
            topk_weights, topk_ids = moe_topk.biased_grouped_topk_gpu(
                hidden_states=torch.randn(gating_output.shape[0], 8),
                gating_output=gating_output,
                correction_bias=correction_bias,
                topk=3,
                renormalize=True,
                num_expert_group=1,
                topk_group=1,
                num_fused_shared_experts=1,
                routed_scaling_factor=2.0,
                apply_routed_scaling_factor_on_output=True,
            )

        mock_grouped_topk.assert_called_once()
        self.assertEqual(mock_grouped_topk.call_args.args[4], 2)
        self.assertTrue(torch.equal(topk_ids[:, :2], fake_ids))
        self.assertTrue(torch.equal(topk_ids[:, 2], torch.full((2,), 4, dtype=torch.int32)))
        self.assertTrue(torch.equal(topk_weights[:, :2], fake_weights))
        expected_shared = fake_weights.sum(dim=-1) / 2.0
        self.assertTrue(torch.allclose(topk_weights[:, 2], expected_shared))


if __name__ == "__main__":
    unittest.main()
