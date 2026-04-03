"""Nightly performance benchmark for MiniMax-M2.5 on MI325/MI300X (8-GPU).

This test benchmarks MiniMax-M2.5 with TP=8 + EP=8 configuration.

The model path can be configured via MINIMAX_M25_MODEL_PATH environment variable.

Registry: nightly-perf-8-gpu-minimax-m25 suite

Example usage:
    python -m pytest test_minimax_m25_perf_amd.py -v
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import generate_simple_markdown_report
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

register_amd_ci(est_time=5400, suite="nightly-perf-8-gpu-minimax-m25", nightly=True)


MINIMAX_M25_MODEL_PATH = os.environ.get(
    "MINIMAX_M25_MODEL_PATH", "MiniMaxAI/MiniMax-M2.5"
)
PROFILE_DIR = "performance_profiles_minimax_m25"


class TestNightlyMiniMaxM25Performance(unittest.TestCase):
    """Nightly performance benchmark for MiniMax-M2.5 on MI325/MI300X.

    Tests MiniMax-M2.5 with TP=8 + EP=8 configuration.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        cls.model_config = {
            "name": "minimax-m25-tp8-ep8",
            "model_path": MINIMAX_M25_MODEL_PATH,
            "other_args": [
                "--trust-remote-code",
                "--tp",
                "8",
                "--ep-size",
                "8",
                "--attention-backend",
                "aiter",
                "--mem-fraction-static",
                "0.85",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
                "--watchdog-timeout",
                "1200",
            ],
            "env_vars": {
                "SGLANG_USE_AITER": "1",
            },
        }

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_minimax_m25(self):
        """Run benchmark for MiniMax-M2.5."""
        old_env = {}
        for key, value in self.model_config.get("env_vars", {}).items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
            print(f"Setting env: {key}={value}")

        try:
            result_tuple = self.runner.run_benchmark_for_model(
                model_path=self.model_config["model_path"],
                batch_sizes=self.batch_sizes,
                input_lens=self.input_lens,
                output_lens=self.output_lens,
                other_args=self.model_config["other_args"],
                variant=self.model_config["name"],
                extra_bench_args=["--trust-remote-code"],
                enable_profile=False,
                timeout=5400,
            )
            results = result_tuple[0]
            success = result_tuple[1]

            if results:
                self.runner.full_report += (
                    generate_simple_markdown_report(results) + "\n"
                )

            self.assertTrue(success, "Benchmark failed for MiniMax-M2.5")
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
