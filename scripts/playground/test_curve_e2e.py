#!/usr/bin/env python3
"""
E2E test for CurveZMQ auto-gen on a real SGLang server.

Tests:
  1. Server starts with auto-generated CURVE keys (log line present).
  2. Internal ZMQ (scheduler, tokenizer-manager, detokenizer-manager) works through CURVE.
  3. HTTP inference requests succeed normally.
  4. --no-zmq-curve disables CURVE and server still works.
  5. SGLANG_ZMQ_CURVE_PUBLIC/SECRET_KEY env override is accepted.

Usage:
    uv run python scripts/playground/test_curve_e2e.py
    uv run python scripts/playground/test_curve_e2e.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import fcntl
import os
import select
import subprocess
import sys
import time

import requests

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_PORT = 32100
LAUNCH_TIMEOUT = 360  # seconds (multi-GPU tests need more time)
TEST_PROMPT = "Hello, what is 1+1?"

SGLANG_CMD = [sys.executable, "-m", "sglang.launch_server"]


def _launch(port, extra_args=(), env=None):
    cmd = SGLANG_CMD + [
        "--model-path", MODEL,
        "--port", str(port),
        "--tp", "1",
        "--host", "127.0.0.1",
    ] + list(extra_args)
    print(f"\n[launch] {' '.join(cmd)}")
    # Force HF offline mode so each GPU worker doesn't retry slow network calls
    # for a model that's already in the local cache.
    offline = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    merged_env = {**os.environ, **offline, **(env or {})}
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
    )


def _wait_ready(port, proc, timeout=LAUNCH_TIMEOUT):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    output_lines = []

    # Make stdout non-blocking so readline() never deadlocks when the server
    # stops emitting log lines (e.g. after CUDA graph capture finishes).
    fd = proc.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    buf = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            remaining = proc.stdout.read() or ""
            output_lines.extend(remaining.splitlines())
            raise RuntimeError(
                f"Server exited early (code {proc.returncode}):\n"
                + "\n".join(output_lines[-40:])
            )

        # Check health first so we exit as soon as the server is ready.
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return output_lines
        except Exception:
            pass

        # Drain whatever the server has printed without blocking.
        ready, _, _ = select.select([proc.stdout], [], [], 0.5)
        if ready:
            chunk = proc.stdout.read(65536) or ""
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                output_lines.append(line)
                if "GET /health" not in line:
                    print(line)

    raise RuntimeError(f"Server on port {port} not ready after {timeout}s")


def _collect_remaining(proc, timeout=5):
    """Drain any remaining buffered output (fd is already O_NONBLOCK)."""
    lines = []
    deadline = time.time() + timeout
    buf = ""
    while time.time() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 0.2)
        if not ready:
            break
        chunk = proc.stdout.read(65536) or ""
        if not chunk:
            break
        buf += chunk
    for line in buf.splitlines():
        lines.append(line)
        print(line)
    return lines


def _kill(proc):
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def _infer(port):
    resp = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={"text": TEST_PROMPT, "sampling_params": {"max_new_tokens": 16}},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    text = result.get("text", "")
    print(f"  response: {text!r}")
    return text


def test_auto_gen_curve(port):
    print("\n" + "=" * 60)
    print("TEST 1: Auto-generated CurveZMQ keypair")
    print("=" * 60)
    proc = _launch(port)
    try:
        lines = _wait_ready(port, proc)
        lines += _collect_remaining(proc)
        all_output = "\n".join(lines)

        assert "CurveZMQ: enabled" in all_output, (
            "Expected 'CurveZMQ: enabled' log line in server output, not found.\n"
            f"Last output:\n{all_output[-2000:]}"
        )
        print("  [OK] Found 'CurveZMQ: enabled' in server logs")

        text = _infer(port)
        assert len(text) > 0, "Empty inference response"
        print("  [OK] Inference succeeded with CURVE enabled")
    finally:
        _kill(proc)


def test_tp2_curve(port):
    print("\n" + "=" * 60)
    print("TEST 2: TP=2 -- scheduler→worker ZMQ through CURVE")
    print("=" * 60)
    proc = _launch(port, extra_args=["--tp", "2"])
    try:
        lines = _wait_ready(port, proc)
        lines += _collect_remaining(proc)
        all_output = "\n".join(lines)

        assert "CurveZMQ: enabled" in all_output, (
            "Expected 'CurveZMQ: enabled' log with TP=2"
        )
        print("  [OK] Found 'CurveZMQ: enabled' with TP=2")

        text = _infer(port)
        assert len(text) > 0, "Empty inference response"
        print("  [OK] Inference succeeded with TP=2 + CURVE")
    finally:
        _kill(proc)


def test_tp2_dp2_curve(port, timeout=LAUNCH_TIMEOUT):
    print("\n" + "=" * 60)
    print("TEST 3: TP=2 DP=2 -- DP controller ZMQ through CURVE (4 GPUs)")
    print("=" * 60)
    proc = _launch(port, extra_args=["--tp", "2", "--dp", "2"])
    try:
        lines = _wait_ready(port, proc, timeout=timeout)
        lines += _collect_remaining(proc)
        all_output = "\n".join(lines)

        assert "CurveZMQ: enabled" in all_output, (
            "Expected 'CurveZMQ: enabled' log with TP=2 DP=2"
        )
        print("  [OK] Found 'CurveZMQ: enabled' with TP=2 DP=2")

        text = _infer(port)
        assert len(text) > 0, "Empty inference response"
        print("  [OK] Inference succeeded with TP=2 DP=2 + CURVE")
    finally:
        _kill(proc)


def test_no_zmq_curve(port):
    print("\n" + "=" * 60)
    print("TEST 4: --no-zmq-curve disables CURVE")
    print("=" * 60)
    proc = _launch(port, extra_args=["--no-zmq-curve"])
    try:
        lines = _wait_ready(port, proc)
        lines += _collect_remaining(proc)
        all_output = "\n".join(lines)

        assert "CurveZMQ: disabled" in all_output, (
            "Expected 'CurveZMQ: disabled' log with --no-zmq-curve"
        )
        assert "CurveZMQ: enabled" not in all_output, (
            "CURVE should be disabled but found enabled log"
        )
        print("  [OK] Found 'CurveZMQ: disabled' log (CURVE disabled)")

        text = _infer(port)
        assert len(text) > 0, "Empty inference response"
        print("  [OK] Inference succeeded with CURVE disabled")
    finally:
        _kill(proc)


def test_env_var_keys(port):
    print("\n" + "=" * 60)
    print("TEST 5: SGLANG_ZMQ_CURVE_PUBLIC/SECRET_KEY env override")
    print("=" * 60)
    import zmq
    if not zmq.has("curve"):
        print("  [SKIP] libzmq built without CURVE support")
        return

    pub, sec = zmq.curve_keypair()
    env = {
        "SGLANG_ZMQ_CURVE_PUBLIC_KEY": pub.decode("ascii"),
        "SGLANG_ZMQ_CURVE_SECRET_KEY": sec.decode("ascii"),
    }
    proc = _launch(port, env=env)
    try:
        lines = _wait_ready(port, proc)
        lines += _collect_remaining(proc)
        all_output = "\n".join(lines)

        assert "CurveZMQ: using keypair from environment variables" in all_output, (
            "Expected env var key log line"
        )
        print("  [OK] Server used env var keys")

        text = _infer(port)
        assert len(text) > 0, "Empty inference response"
        print("  [OK] Inference succeeded with env var CURVE keys")
    finally:
        _kill(proc)


def main():
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--test",
        action="append",
        metavar="NAME",
        help="Run only this test (may be repeated). "
             "Choices: auto_gen_curve, tp2_curve, tp2_dp2_curve, no_zmq_curve, env_var_keys",
    )
    args = parser.parse_args()
    MODEL = args.model

    port = BASE_PORT
    passed = []
    failed = []

    all_tests = [
        ("auto_gen_curve",  test_auto_gen_curve,  port),
        ("tp2_curve",       test_tp2_curve,       port + 1),
        ("tp2_dp2_curve",   test_tp2_dp2_curve,   port + 2),
        ("no_zmq_curve",    test_no_zmq_curve,    port + 3),
        ("env_var_keys",    test_env_var_keys,    port + 4),
    ]
    tests = [(n, f, p) for n, f, p in all_tests if not args.test or n in args.test]

    for name, fn, p in tests:
        try:
            fn(p)
            passed.append(name)
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if passed:
        print(f"  PASSED: {', '.join(passed)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
