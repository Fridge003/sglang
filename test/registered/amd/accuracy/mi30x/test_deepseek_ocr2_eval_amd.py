"""DeepSeek-OCR-2 MMMU Evaluation Test (1-GPU)

Tests DeepSeek-OCR-2 on the MMMU benchmark on AMD GPUs.
Uses triton attention (aiter pa_ragged JIT compilation OOMs via hipcc).

Registry: nightly-amd-accuracy-1-gpu-deepseek-ocr2 suite
"""

import os
import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=600, suite="nightly-amd-accuracy-2-gpu-vlm", nightly=True)

MODEL_PATH = "deepseek-ai/DeepSeek-OCR-2"
ACCURACY_THRESHOLD = 0.25
NUM_MMMU_SAMPLES = 100


class TestDeepSeekOCR2EvalAMD(CustomTestCase):
    """DeepSeek-OCR-2 MMMU Evaluation Test for AMD."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmmu_accuracy(self):
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "0"

        other_args = [
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.70",
            "--max-total-tokens",
            "16384",
        ]

        print(f"\nTesting: {MODEL_PATH}")
        server_start = time.time()
        process = popen_launch_server(
            model=MODEL_PATH,
            base_url=self.base_url,
            other_args=other_args,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
        )
        startup_time = time.time() - server_start
        print(f"Server startup: {startup_time:.1f}s")

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=MODEL_PATH,
                eval_name="mmmu",
                num_examples=NUM_MMMU_SAMPLES,
                num_threads=64,
                max_tokens=30,
            )

            eval_start = time.time()
            metrics = run_eval(args)
            eval_time = time.time() - eval_start
            score = metrics["score"]

            print(f"Score: {score:.3f} (threshold: {ACCURACY_THRESHOLD})")
            print(f"Eval time: {eval_time:.1f}s")

            if is_in_ci():
                status = "PASS" if score >= ACCURACY_THRESHOLD else "FAIL"
                write_github_step_summary(
                    f"### DeepSeek-OCR-2 MMMU\n"
                    f"| Model | Score | Threshold | Status |\n"
                    f"| ----- | ----- | --------- | ------ |\n"
                    f"| {MODEL_PATH} | {score:.3f} | {ACCURACY_THRESHOLD} | {status} |\n"
                )

            self.assertGreaterEqual(
                score,
                ACCURACY_THRESHOLD,
                f"MMMU score {score:.3f} below threshold {ACCURACY_THRESHOLD}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
