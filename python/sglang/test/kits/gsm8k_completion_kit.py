"""Shared helpers for GSM8K completion-style accuracy tests."""

from types import SimpleNamespace
from typing import Tuple
from urllib.parse import urlparse

from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k


def _split_base_url(base_url: str) -> tuple[str, int]:
    """Split a runtime base URL into the host and port run_eval expects."""
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.hostname or parsed.port is None:
        raise ValueError(
            f"base_url must include scheme, host, and port; got {base_url!r}"
        )

    return f"{parsed.scheme}://{parsed.hostname}", parsed.port


def run_gsm8k_benchmark(
    base_url: str,
    num_questions: int = 200,
    num_shots: int = 5,
    parallel: int = 64,
) -> Tuple[float, float, float]:
    """Run the shared few-shot GSM8K completion benchmark."""
    host, port = _split_base_url(base_url)

    metrics = run_eval_gsm8k(
        SimpleNamespace(
            num_shots=num_shots,
            data_path=None,
            num_questions=num_questions,
            max_new_tokens=512,
            parallel=parallel,
            host=host,
            port=port,
        )
    )

    return (
        float(metrics["accuracy"]),
        float(metrics["invalid"]),
        float(metrics["latency"]),
    )
