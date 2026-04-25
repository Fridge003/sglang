import torch
from unittest.mock import patch

from sglang.jit_kernel import flash_attention_v3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class TestFlashAttentionV3Compatibility(CustomTestCase):
    def _kvcache_inputs(self):
        q = torch.zeros((1, 1, 1, 8), dtype=torch.float16)
        k_cache = torch.zeros((1, 1, 1, 8), dtype=torch.float16)
        v_cache = torch.zeros((1, 1, 1, 8), dtype=torch.float16)
        return q, k_cache, v_cache

    def _varlen_inputs(self):
        q = torch.zeros((1, 1, 8), dtype=torch.float16)
        k = torch.zeros((1, 1, 8), dtype=torch.float16)
        v = torch.zeros((1, 1, 8), dtype=torch.float16)
        cu = torch.tensor([0, 1], dtype=torch.int32)
        return q, k, v, cu, cu

    def test_kvcache_omits_out_for_legacy_kernel_signature(self):
        captured = {}

        def kernel(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "ok"

        with (
            patch.object(flash_attention_v3, "_is_fa3_supported", return_value=True),
            patch.object(
                flash_attention_v3,
                "_load_fa3_kernels",
                return_value={"flash_attn_with_kvcache": kernel},
            ),
        ):
            result = flash_attention_v3.flash_attn_with_kvcache(
                *self._kvcache_inputs(),
                out=torch.ones((1, 1, 1, 8), dtype=torch.float16),
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["kwargs"], {})

    def test_kvcache_forwards_out_when_kernel_supports_it(self):
        captured = {}

        def kernel(*args, out=None):
            captured["out"] = out
            return "ok"

        expected_out = torch.ones((1, 1, 1, 8), dtype=torch.float16)
        with (
            patch.object(flash_attention_v3, "_is_fa3_supported", return_value=True),
            patch.object(
                flash_attention_v3,
                "_load_fa3_kernels",
                return_value={"flash_attn_with_kvcache": kernel},
            ),
        ):
            result = flash_attention_v3.flash_attn_with_kvcache(
                *self._kvcache_inputs(),
                out=expected_out,
            )

        self.assertEqual(result, "ok")
        self.assertIs(captured["out"], expected_out)

    def test_varlen_omits_out_for_legacy_kernel_signature(self):
        captured = {}

        def kernel(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "ok"

        with (
            patch.object(flash_attention_v3, "_is_fa3_supported", return_value=True),
            patch.object(
                flash_attention_v3,
                "_load_fa3_kernels",
                return_value={"flash_attn_varlen_func": kernel},
            ),
        ):
            result = flash_attention_v3.flash_attn_varlen_func(
                *self._varlen_inputs(),
                out=torch.ones((1, 1, 8), dtype=torch.float16),
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["kwargs"], {})

    def test_varlen_forwards_out_when_kernel_supports_it(self):
        captured = {}

        def kernel(*args, out=None):
            captured["out"] = out
            return "ok"

        expected_out = torch.ones((1, 1, 8), dtype=torch.float16)
        with (
            patch.object(flash_attention_v3, "_is_fa3_supported", return_value=True),
            patch.object(
                flash_attention_v3,
                "_load_fa3_kernels",
                return_value={"flash_attn_varlen_func": kernel},
            ),
        ):
            result = flash_attention_v3.flash_attn_varlen_func(
                *self._varlen_inputs(),
                out=expected_out,
            )

        self.assertEqual(result, "ok")
        self.assertIs(captured["out"], expected_out)


if __name__ == "__main__":
    import unittest

    unittest.main()
