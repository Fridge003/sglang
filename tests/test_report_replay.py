#!/usr/bin/env python3
"""Behavior tests for the standalone replay report command."""

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import report_replay  # noqa: E402


BASE_TIME = datetime(2026, 8, 14, tzinfo=timezone.utc)


def make_row(
    request_id,
    start_s,
    end_s,
    *,
    ttft_s=0.1,
    tpot_s=0.2,
    prompt=100,
    output=3,
    cached=0,
    success=True,
):
    return {
        "request_id": request_id,
        "completed_at": (BASE_TIME + timedelta(seconds=end_s)).isoformat(),
        "latency_s": end_s - start_s,
        "ttft_s": ttft_s,
        "tpot_s": tpot_s,
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "cached_tokens": cached,
        "success": success,
    }


class RequestMetricTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(report_replay.percentile([0, 10], 50), 5)
        self.assertEqual(report_replay.percentile([0, 10], 99), 9.9)

    def test_request_metrics_use_decode_intervals_and_actual_cache_usage(self):
        rows = [
            make_row(
                "a",
                0,
                4,
                ttft_s=1,
                tpot_s=0.1,
                prompt=100,
                output=7,
                cached=80,
            ),
            make_row(
                "b",
                1,
                3,
                ttft_s=0.5,
                tpot_s=0.2,
                prompt=200,
                output=6,
                cached=100,
            ),
        ]

        got = report_replay.request_metric_summary(rows)

        self.assertEqual(got["p50_ttft_ms"], 750.0)
        self.assertEqual(got["p99_ttft_ms"], 995.0)
        self.assertAlmostEqual(got["p50_tpot_ms"], 150.0)
        self.assertAlmostEqual(got["p99_tpot_ms"], 199.0)
        self.assertAlmostEqual(got["p50_decode_tps_per_user"], 8 / 3)
        self.assertEqual(got["p50_cache_hit_rate"], 0.65)
        self.assertEqual(got["p99_cache_hit_rate"], 0.797)

    def test_load_requests_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requests.jsonl"
            row = make_row("duplicate", 0, 1)
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate request_id"):
                report_replay.load_requests(path)


class LoadMetricTest(unittest.TestCase):
    def test_concurrency_distribution_includes_idle_gaps(self):
        rows = [
            make_row("a", 0, 1),
            make_row("b", 3, 4),
        ]

        got = report_replay.concurrency_summary(rows)

        self.assertEqual(got["p50_concurrency"], 0)
        self.assertEqual(got["p99_concurrency"], 1)
        self.assertEqual(got["mean_concurrency"], 0.5)
        self.assertEqual(got["duration_s"], 4.0)

    def test_qps_uses_arrival_windows_and_includes_empty_windows(self):
        rows = [
            make_row("a", 0, 0.5),
            make_row("b", 0.2, 0.4),
            make_row("c", 2.4, 4),
        ]

        got = report_replay.qps_summary(rows, 1.0)

        self.assertEqual(got["window_count"], 4)
        self.assertEqual(got["p50_qps"], 0.5)
        self.assertAlmostEqual(got["p99_qps"], 1.97)


class AcceptanceHistogramTest(unittest.TestCase):
    def test_histogram_uses_inclusive_power_of_two_upper_bounds(self):
        got = report_replay.build_histogram([0, 1, 128, 129, 256, 257], [128, 256])

        self.assertEqual([bucket["count"] for bucket in got], [3, 2, 1])
        self.assertEqual(
            [bucket["range"] for bucket in got],
            ["0-128", "129-256", ">256"],
        )
        self.assertEqual(got[-1]["cumulative_percentage"], 100.0)

    def test_acceptance_samples_are_limited_to_formal_replay_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "server.log"
            path.write_text(
                "\n".join(
                    [
                        "[2026-08-13 23:59:59 TP0] Decode batch, accept len: 9.0, accept rate: 0.9",
                        "[2026-08-14 00:00:01 TP0] Decode batch, accept len: 2.5, accept rate: 0.2",
                        "[2026-08-14 00:00:03 TP0] Decode batch, accept len: 4.0, accept rate: 0.4",
                        "[2026-08-14 00:00:05 TP0] Decode batch, accept len: 8.0, accept rate: 0.8",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            got = report_replay.parse_acceptance_samples(
                path, BASE_TIME, BASE_TIME + timedelta(seconds=4)
            )

        self.assertEqual(got, [2.5, 4.0])


class CliTest(unittest.TestCase):
    def test_cli_writes_complete_markdown_and_json_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            requests_path = temp / "requests.jsonl"
            summary_path = temp / "summary.json"
            server_log_path = temp / "server.log"
            output_dir = temp / "report"
            rows = [
                make_row(
                    "a",
                    0,
                    4,
                    ttft_s=1,
                    tpot_s=0.1,
                    prompt=100,
                    output=7,
                    cached=80,
                ),
                make_row(
                    "b",
                    1,
                    3,
                    ttft_s=0.5,
                    tpot_s=0.2,
                    prompt=200,
                    output=6,
                    cached=100,
                ),
            ]
            requests_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "total_requests": 2,
                        "successful_requests": 2,
                        "failed_requests": 0,
                        "replay_duration_s": 4.0,
                        "decode_tps": 3.25,
                        "mean_accept_length": 3.5,
                    }
                ),
                encoding="utf-8",
            )
            server_log_path.write_text(
                "\n".join(
                    [
                        "[2026-08-14 00:00:01 TP0] Decode batch, accept len: 2.0, accept rate: 0.2",
                        "[2026-08-14 00:00:03 TP0] Decode batch, accept len: 4.0, accept rate: 0.4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "report_replay.py"),
                    "--requests",
                    str(requests_path),
                    "--summary",
                    str(summary_path),
                    "--server-log",
                    str(server_log_path),
                    "--output-dir",
                    str(output_dir),
                    "--qps-window-s",
                    "1",
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output_dir / "report.json").read_text())
            markdown = (output_dir / "report.md").read_text()

        self.assertEqual(report["counts"]["total_requests"], 2)
        self.assertEqual(report["counts"]["successful_requests"], 2)
        self.assertEqual(report["metrics"]["p50_ttft_ms"], 750.0)
        self.assertAlmostEqual(report["metrics"]["p50_decode_tps_per_user"], 8 / 3)
        self.assertEqual(report["acceptance"]["p50"], 3.0)
        self.assertEqual(report["acceptance"]["p99"], 3.98)
        self.assertEqual(report["acceptance"]["mean"], 3.5)
        self.assertEqual(report["acceptance"]["mean_source"], "summary")
        self.assertEqual(sum(x["count"] for x in report["isl_histogram"]), 2)
        self.assertEqual(sum(x["count"] for x in report["osl_histogram"]), 2)
        self.assertIn("P50 Decode TPS per user", markdown)
        self.assertIn("ISL Histogram", markdown)
        self.assertIn("OSL Histogram", markdown)
        self.assertIn("Source artifacts", markdown)
        self.assertNotIn("ITL", markdown)


if __name__ == "__main__":
    unittest.main()
