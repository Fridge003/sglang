"""AMD Nightly performance benchmark for DeepSeek-V3.2 model (basic variant).

This test benchmarks the DeepSeek-V3.2 model with basic TP=8 configuration on 8 GPUs.

The model path can be configured via DEEPSEEK_V32_MODEL_PATH environment variable.

Registry: nightly-perf-8-gpu-deepseek-v32-basic suite

Example usage:
    DEEPSEEK_V32_MODEL_PATH=deepseek-ai/DeepSeek-V3.2 python -m pytest test_deepseek_v32_basic_perf_amd.py -v
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import generate_simple_markdown_report
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register for AMD CI - DeepSeek-V3.2 basic benchmark (~90 min)
register_amd_ci(
    est_time=5400, suite="nightly-perf-8-gpu-deepseek-v32-basic", nightly=True
)


# Model path can be overridden via environment variable
DEEPSEEK_V32_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V32_MODEL_PATH", "deepseek-ai/DeepSeek-V3.2"
)
PROFILE_DIR = "performance_profiles_deepseek_v32_basic_mi325"


class TestNightlyDeepseekV32BasicPerformance(unittest.TestCase):
    """AMD Nightly performance benchmark for DeepSeek-V3.2 model (basic variant).

    Tests the DeepSeek-V3.2 model with basic TP=8 configuration on MI325/MI300X.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Basic variant configuration for DeepSeek-V3.2
        # MI325 uses aiter attention backend
        cls.variant_config = {
            "name": "basic",
            "other_args": [
                "--trust-remote-code",
                "--tp",
                "8",
                "--attention-backend",
                "aiter",
                "--chunked-prefill-size",
                "131072",
                "--mem-fraction-static",
                "0.85",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
                "--watchdog-timeout",
                "1200",
            ],
            "env_vars": {"SGLANG_USE_AITER": "1"},
        }

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        # Override full_report to remove traces help text
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Run benchmark for basic variant."""
        try:
            result_tuple = self.runner.run_benchmark_for_model(
                model_path=self.model,
                batch_sizes=self.batch_sizes,
                input_lens=self.input_lens,
                output_lens=self.output_lens,
                other_args=self.variant_config["other_args"],
                variant=self.variant_config["name"],
                extra_bench_args=["--trust-remote-code"],
                enable_profile=False,  # Disable profiling for AMD tests
            )
            results = result_tuple[0]
            success = result_tuple[1]
            avg_spec_accept_length = result_tuple[2] if len(result_tuple) > 2 else None

            # Log speculative decoding accept length
            if avg_spec_accept_length is not None:
                print(f"  avg_spec_accept_length={avg_spec_accept_length:.2f}")

            # Use simplified report format without traces
            if results:
                self.runner.full_report += (
                    generate_simple_markdown_report(results) + "\n"
                )

            if not success:
                raise AssertionError(
                    f"Benchmark failed for {self.model} (basic variant)"
                )
        finally:
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
