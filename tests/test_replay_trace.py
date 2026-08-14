import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from replay_trace import (
    StreamAccumulator,
    build_payload,
    build_replay_requests,
    cache_hit_rate,
    feed_sse_bytes,
    normalize_cached_tokens,
    percentile,
    summarize_records,
)


def synthetic_sequence():
    return {
        "sequence_id": "seq-1",
        "tools": [
            {
                "name": "bash",
                "description": "Run a command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                "strict": False,
            }
        ],
        "events": [
            {
                "kind": "system_message",
                "content": {"parts": [{"text": "system"}]},
            },
            {
                "kind": "user_message",
                "content": {"parts": [{"text": "question"}]},
            },
            {
                "kind": "agent_message",
                "content": {"parts": [{"text": "visible"}]},
                "reasoning": "think",
                "tool_calls": [
                    {
                        "call_id": "call-a",
                        "name": "bash",
                        "arguments": {"command": "first"},
                    },
                    {
                        "call_id": "call-b",
                        "name": "bash",
                        "arguments": {"command": "second"},
                    },
                ],
                "extra": {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 7,
                        "cached_tokens": 0,
                    }
                },
            },
            {
                "kind": "tool_result",
                "call_ref": {"sequence": "seq-1", "index": 2},
                "call_index": 0,
                "result": {
                    "status": "success",
                    "content": {"parts": [{"text": "first result"}]},
                    "duration_ms": 100,
                },
            },
            {
                "kind": "tool_result",
                "call_ref": {"sequence": "seq-1", "index": 2},
                "call_index": 1,
                "result": {
                    "status": "success",
                    "content": {"parts": [{"text": "second result"}]},
                    "duration_ms": 200,
                },
            },
            {
                "kind": "agent_message",
                "content": {"parts": [{"text": "done"}]},
                "reasoning": "final thought",
                "tool_calls": [],
                "extra": {
                    "usage": {
                        "prompt_tokens": 25,
                        "completion_tokens": 3,
                        "cached_tokens": 8,
                    }
                },
            },
        ],
    }


class ReplayRequestTest(unittest.TestCase):
    def test_reconstructs_history_and_sums_tool_think_time(self):
        """Dropping reasoning/tool IDs or one tool duration must fail."""
        requests = build_replay_requests(
            synthetic_sequence(), trajectory_id="traj-1", start_offset_s=1.5
        )

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["max_tokens"], 7)
        self.assertEqual(requests[0]["start_offset_s"], 1.5)
        self.assertEqual(requests[0]["think_time_before_s"], 0.0)
        self.assertEqual(
            requests[0]["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
        )

        second = requests[1]
        self.assertAlmostEqual(second["think_time_before_s"], 0.3)
        self.assertEqual(second["max_tokens"], 3)
        self.assertEqual(second["trace_prompt_tokens"], 25)
        assistant = second["messages"][2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"], "visible")
        self.assertEqual(assistant["reasoning_content"], "think")
        self.assertEqual(assistant["tool_calls"][1]["id"], "call-b")
        self.assertEqual(
            [message["tool_call_id"] for message in second["messages"][3:]],
            ["call-a", "call-b"],
        )
        self.assertEqual(second["tools"][0]["function"]["name"], "bash")
        self.assertEqual(
            second["tools"][0]["function"]["parameters"]["required"],
            ["command"],
        )


class MetricTest(unittest.TestCase):
    def test_missing_cache_details_means_zero_when_reporting_is_enabled(self):
        """SGLang omits prompt_tokens_details when the cached count is zero."""
        self.assertEqual(normalize_cached_tokens(None, 100, True), 0)
        self.assertEqual(normalize_cached_tokens(64, 100, True), 64)
        self.assertIsNone(normalize_cached_tokens(None, 100, False))

    def test_percentile_uses_linear_interpolation(self):
        """Nearest-rank percentile substitution must fail this test."""
        self.assertEqual(percentile([0.0, 10.0], 50), 5.0)
        self.assertEqual(percentile([0.0, 10.0], 99), 9.9)

    def test_cache_hit_rate_uses_actual_prompt_tokens(self):
        """Using trace prompt tokens instead of server usage must fail."""
        self.assertEqual(cache_hit_rate(64, 80), 0.8)
        self.assertIsNone(cache_hit_rate(None, 80))
        self.assertIsNone(cache_hit_rate(0, 0))

    def test_stream_accumulator_counts_reasoning_as_first_token(self):
        """Waiting for visible content would overstate TTFT."""
        acc = StreamAccumulator()
        acc.update(
            {"choices": [{"delta": {"reasoning_content": "r"}}]},
            elapsed_s=0.2,
        )
        acc.update(
            {"choices": [{"delta": {"content": "answer"}}]}, elapsed_s=0.4
        )
        acc.update(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 64},
                },
            },
            elapsed_s=0.6,
        )
        result = acc.finish(total_latency_s=0.6)
        self.assertAlmostEqual(result["ttft_s"], 0.2)
        self.assertAlmostEqual(result["tpot_s"], 0.2)
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["cached_tokens"], 64)

    def test_stream_accumulator_counts_tool_call_delta_as_first_token(self):
        """Tool-only responses must still receive a TTFT."""
        acc = StreamAccumulator()
        acc.update(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": "{"}}
                            ]
                        }
                    }
                ]
            },
            elapsed_s=0.1,
        )
        self.assertEqual(acc.finish(0.3)["ttft_s"], 0.1)

    def test_sse_parser_handles_split_frames_and_done(self):
        """Assuming each TCP chunk is one SSE frame must fail."""
        state = bytearray()
        first = b'data: {"choices":[{"delta":{"content":"a"}}]}\n'
        second = b'\ndata: {"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
        payloads = []
        payloads.extend(feed_sse_bytes(state, first[:13]))
        payloads.extend(feed_sse_bytes(state, first[13:] + second[:20]))
        payloads.extend(feed_sse_bytes(state, second[20:]))
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["choices"][0]["delta"]["content"], "a")
        self.assertEqual(payloads[1]["usage"]["completion_tokens"], 1)

    def test_payload_requests_exact_output_and_cache_usage(self):
        """EOS truncation or missing usage/cache details changes the workload."""
        request = build_replay_requests(
            synthetic_sequence(), trajectory_id="traj-1", start_offset_s=0
        )[0]
        payload = build_payload(request, model="model-id", respect_eos=False)
        self.assertEqual(payload["model"], "model-id")
        self.assertEqual(payload["max_tokens"], 7)
        self.assertTrue(payload["ignore_eos"])
        self.assertTrue(payload["stream"])
        self.assertTrue(payload["stream_options"]["include_usage"])
        self.assertTrue(payload["return_cached_tokens_details"])

    def test_summary_excludes_failures_and_uses_replay_wall_time(self):
        """Including failures or per-request time in aggregate TPS must fail."""
        records = [
            {
                "success": True,
                "ttft_s": 0.1,
                "tpot_s": 0.01,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 50,
            },
            {
                "success": True,
                "ttft_s": 0.3,
                "tpot_s": 0.03,
                "prompt_tokens": 200,
                "completion_tokens": 30,
                "cached_tokens": 100,
            },
            {
                "success": False,
                "ttft_s": 99,
                "tpot_s": 99,
                "prompt_tokens": 999,
                "completion_tokens": 999,
                "cached_tokens": 999,
            },
        ]
        summary = summarize_records(
            records, replay_duration_s=10.0, mean_accept_length=4.25
        )
        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["p50_ttft_ms"], 200.0)
        self.assertEqual(summary["p99_ttft_ms"], 298.0)
        self.assertEqual(summary["p50_tpot_ms"], 20.0)
        self.assertEqual(summary["decode_tps"], 5.0)
        self.assertEqual(summary["p50_cache_hit_rate"], 0.5)
        self.assertEqual(summary["token_weighted_cache_hit_rate"], 0.5)
        self.assertEqual(summary["mean_accept_length"], 4.25)


class CliTest(unittest.TestCase):
    def test_server_launch_enables_openai_cache_reporting(self):
        launch_script = (SCRIPTS / "launch_sglang.sh").read_text()
        self.assertIn("--enable-cache-report", launch_script)

    def test_server_launch_uses_second_attempt_memory_fraction_only(self):
        launch_script = (SCRIPTS / "launch_sglang.sh").read_text()
        self.assertIn("--mem-fraction-static 0.85", launch_script)
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", launch_script)

    def test_formal_replay_is_fail_fast(self):
        formal_script = (SCRIPTS / "run_full_replay.sh").read_text()
        self.assertIn("--fail-fast", formal_script)

    def test_runners_locate_replay_script_from_any_working_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$SCRIPT_CAPTURE\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            for runner in ("run_smoke_replay.sh", "run_full_replay.sh"):
                with self.subTest(runner=runner):
                    capture = temp_path / f"{runner}.args"
                    environment["SCRIPT_CAPTURE"] = str(capture)
                    completed = subprocess.run(
                        [
                            str(SCRIPTS / runner),
                            str(temp_path / "trace.zip"),
                            str(temp_path / runner),
                        ],
                        cwd=temp_path,
                        env=environment,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertEqual(
                        capture.read_text(encoding="utf-8").splitlines()[0],
                        str(SCRIPTS / "replay_trace.py"),
                    )

    def test_dry_run_does_not_require_aiohttp(self):
        """Moving the network import above the dry-run exit must fail."""
        trajectory = {
            "trajectory_id": "traj-dry",
            "sequences": {
                "seq-dry": {
                    "sequence_id": "seq-dry",
                    "lifecycle": {"started_at": "2026-08-10T00:00:00Z"},
                    "tools": [],
                    "events": [
                        {
                            "kind": "agent_message",
                            "content": {"parts": [{"text": "x"}]},
                            "extra": {
                                "usage": {
                                    "prompt_tokens": 1,
                                    "completion_tokens": 1,
                                }
                            },
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / "trace.zip"
            with zipfile.ZipFile(str(archive_path), "w") as archive:
                archive.writestr(
                    "session/task/rollout/agent/trajectory.json",
                    json.dumps(trajectory),
                )
            command = [
                sys.executable,
                "-S",
                str(SCRIPTS / "replay_trace.py"),
                "--trace",
                str(archive_path),
                "--output-dir",
                str(temp_path / "output"),
                "--dry-run",
            ]
            completed = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("1 trajectories and 1 requests", completed.stdout)


if __name__ == "__main__":
    unittest.main()
