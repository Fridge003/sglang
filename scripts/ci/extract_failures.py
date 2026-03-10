#!/usr/bin/env python3
"""
CI Failure Extractor - Extracts structured failure data from CI test runs.

This script runs as a post-failure step in GitHub Actions CI jobs.
It reads the structured test results JSON produced by ci_utils.py's
write_test_results_json(), enriches it with CI environment metadata,
and outputs a failure report JSON that is uploaded as a GitHub artifact.

Graceful degradation for log parsing:
  1. Full details from structured JSON (/tmp/sglang_test_results.json)
  2. Partial: parse test file names from captured log output
  3. Minimal: record job name + partition for job-level timeouts

Usage in CI:
    python3 scripts/ci/extract_failures.py \
        --job-name "${{ github.job }}" \
        --suite "stage-b-test-large-1-gpu" \
        --partition-id "${{ matrix.partition }}" \
        --output /tmp/failure_report.json
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_test_results_json(results_path: str) -> dict | None:
    """Read structured test results from ci_utils.py output."""
    try:
        with open(results_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def extract_test_cases_from_log(log_text: str) -> list[dict]:
    """
    Parse pytest/unittest output to extract individual test case failures.

    NOTE: This function is intentionally duplicated from scripts/ci_monitor/log_parser.py
    for CI isolation (no project imports needed). Keep the two copies in sync.

    Looks for patterns like:
      FAIL: test_method (test_module.TestClass)
      ERROR: test_method (test_module.TestClass)
      FAILED test_file.py::TestClass::test_method - AssertionError: ...
    """
    test_cases = []

    # Pattern 1: unittest-style FAIL/ERROR
    unittest_pattern = re.compile(
        r"(?:FAIL|ERROR): (\w+) \(([^)]+)\)\s*\n"
        r"-+\s*\n"
        r"(.*?)(?=\n(?:FAIL|ERROR|OK|------|-{10,}|={10,}|Ran \d+))",
        re.DOTALL,
    )
    for match in unittest_pattern.finditer(log_text):
        test_name = match.group(1)
        test_class_path = match.group(2)
        traceback = match.group(3).strip()

        # Extract error type from traceback
        error_type = "Unknown"
        error_message = ""
        error_match = re.search(
            r"(\w+(?:Error|Exception|Failure)):\s*(.*?)$",
            traceback,
            re.MULTILINE,
        )
        if error_match:
            error_type = error_match.group(1)
            error_message = error_match.group(2).strip()

        # Parse class name from module path like "test_autoround.TestAutoRound"
        parts = test_class_path.rsplit(".", 1)
        class_name = parts[-1] if len(parts) > 1 else test_class_path

        test_cases.append(
            {
                "class_name": class_name,
                "test_name": test_name,
                "error_type": error_type,
                "error_message": error_message[:500],  # Truncate long messages
            }
        )

    # Pattern 2: pytest-style FAILED lines
    pytest_pattern = re.compile(
        r"FAILED\s+(\S+\.py)::(\w+)::(\w+)(?:\s*-\s*(.+))?$", re.MULTILINE
    )
    for match in pytest_pattern.finditer(log_text):
        # match.group(1) is the test file path (captured but not needed here)
        class_name = match.group(2)
        test_name = match.group(3)
        error_info = match.group(4) or ""

        error_type = "Unknown"
        error_message = error_info
        if ":" in error_info:
            parts = error_info.split(":", 1)
            error_type = parts[0].strip()
            error_message = parts[1].strip()

        test_cases.append(
            {
                "class_name": class_name,
                "test_name": test_name,
                "error_type": error_type,
                "error_message": error_message[:500],
            }
        )

    return test_cases


def extract_test_cases_for_file(log_text: str, test_file: str) -> list[dict]:
    """
    Extract test case failures from the log section for a specific test file.

    Finds the log output between Begin/End markers for this file, then parses
    unittest/pytest failure patterns within that section. Falls back to parsing
    the entire log if section markers aren't found.
    """
    file_base = test_file.replace(".py", "").split("/")[-1]

    # Try to find the section for this test file between Begin/End markers
    # The log format is:
    #   .\n.\nBegin (N/M):\npython3 /path/to/test_foo.py\n.\n.\n
    #   ... test output ...
    #   .\n.\nEnd (N/M):\n...
    section_pattern = re.compile(
        r"Begin \(\d+/\d+\):\s*\n\s*python3?\s+\S*"
        + re.escape(file_base)
        + r"\.py\s*\n"
        + r"(.*?)"
        + r"(?:\.\n\.\nEnd \(\d+/\d+\):|$)",
        re.DOTALL,
    )
    match = section_pattern.search(log_text)
    if match:
        return extract_test_cases_from_log(match.group(1))

    # Fallback: parse the entire log
    return extract_test_cases_from_log(log_text)


def find_test_files_from_log(log_text: str) -> list[str]:
    """
    Grep test file names from partial/truncated logs.
    Fallback when structured JSON is unavailable.
    """
    test_files = set()

    # Look for "Begin (N/M):" followed by "python3 /path/to/test.py"
    begin_pattern = re.compile(
        r"Begin \(\d+/\d+\):\s*\n\s*python3?\s+(\S+\.py)", re.MULTILINE
    )
    for match in begin_pattern.finditer(log_text):
        path = match.group(1)
        test_files.add(path.split("/")[-1])

    # Look for FAILED: section
    failed_section = re.search(r"FAILED:\s*\n(.*?)(?:={10,}|$)", log_text, re.DOTALL)
    if failed_section:
        for match in re.finditer(r"(\S+\.py)", failed_section.group(1)):
            test_files.add(match.group(1).split("/")[-1])

    return list(test_files)


def find_last_running_test(log_text: str) -> str | None:
    """
    For timeout scenarios, find the last test that was running.
    Looks for the last 'Begin (N/M):' marker or 'server_args:' pattern.
    """
    # Strip ANSI escape codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    log_text = ansi_escape.sub("", log_text)

    lines = log_text.split("\n")

    # Search from bottom for python3 test invocation
    for i in range(len(lines) - 1, -1, -1):
        match = re.search(r"python3?\s+(\S+\.py)", lines[i])
        if match:
            path = match.group(1)
            return path.split("/")[-1]

    return None


def classify_failure(error_type: str, error_message: str) -> str:
    """Classify a failure into a category."""
    combined = f"{error_type} {error_message}".lower()

    if any(
        kw in combined
        for kw in ["accuracy", "score", "not greater than", "not less than"]
    ):
        return "accuracy"
    if any(kw in combined for kw in ["timeout", "timed out"]):
        return "timeout"
    if any(kw in combined for kw in ["oom", "out of memory", "cuda out of memory"]):
        return "oom"
    if any(
        kw in combined
        for kw in ["connectionrefused", "connection refused", "ping failed"]
    ):
        return "connection"
    if any(kw in combined for kw in ["importerror", "modulenotfound"]):
        return "import"
    if any(
        kw in combined for kw in ["segmentation fault", "core dumped", "illegal memory"]
    ):
        return "crash"
    if "assertionerror" in combined or "asserterror" in combined:
        return "assertion"
    return "other"


def build_failure_report(
    test_results: dict | None,
    log_text: str | None,
    job_name: str,
    suite: str,
    partition_id: str | None,
    partition_size: str | None,
) -> dict:
    """Build the structured failure report."""
    report = {
        "schema_version": "1.0",
        "job_name": job_name,
        "workflow": os.environ.get("GITHUB_WORKFLOW", "unknown"),
        "run_id": int(os.environ.get("GITHUB_RUN_ID", 0)),
        "run_number": int(os.environ.get("GITHUB_RUN_NUMBER", 0)),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", 1)),
        "event": os.environ.get("GITHUB_EVENT_NAME", "unknown"),
        "branch": os.environ.get("GITHUB_REF_NAME", "unknown"),
        "head_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "pr_number": None,
        "runner_name": os.environ.get("RUNNER_NAME", "unknown"),
        "suite": suite,
        "partition_id": partition_id,
        "partition_size": partition_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "unknown",
        "failed_tests": [],
        "retried_tests": [],
        "log_truncated": False,
        "overall_passed": 0,
        "overall_total": 0,
    }

    # Try to get PR number from GITHUB_REF (refs/pull/123/merge)
    ref = os.environ.get("GITHUB_REF", "")
    pr_match = re.match(r"refs/pull/(\d+)/merge", ref)
    if pr_match:
        report["pr_number"] = int(pr_match.group(1))

    # Source 1: Structured JSON from ci_utils.py (best quality)
    if test_results:
        report["source"] = "structured_json"
        report["overall_passed"] = test_results.get("passed_count", 0)
        report["overall_total"] = test_results.get("total_tests", 0)

        for failed in test_results.get("failed_tests", []):
            test_file = failed["test_file"]
            failure_reason = failed.get("failure_reason", "unknown")

            # Extract test case failures from the log section for this file
            test_cases = []
            if log_text:
                test_cases = extract_test_cases_for_file(log_text, test_file)

            failure_category = "timeout" if "timeout" in failure_reason else "other"
            if test_cases:
                failure_category = classify_failure(
                    test_cases[0].get("error_type", ""),
                    test_cases[0].get("error_message", ""),
                )

            report["failed_tests"].append(
                {
                    "test_file": test_file.split("/")[-1],
                    "failure_reason": failure_reason,
                    "failure_category": failure_category,
                    "test_cases": test_cases,
                }
            )

        for retried in test_results.get("retried_tests", []):
            report["retried_tests"].append(
                {
                    "test_file": retried["test_file"].split("/")[-1],
                    "attempts": retried["attempts"],
                    "result": retried["result"],
                }
            )

        return report

    # Source 2: Parse log text (fallback)
    if log_text:
        failed_files = find_test_files_from_log(log_text)
        if failed_files:
            report["source"] = "log_parsing"
            for tf in failed_files:
                relevant_cases = extract_test_cases_for_file(log_text, tf)
                failure_category = "other"
                if relevant_cases:
                    failure_category = classify_failure(
                        relevant_cases[0].get("error_type", ""),
                        relevant_cases[0].get("error_message", ""),
                    )
                report["failed_tests"].append(
                    {
                        "test_file": tf,
                        "failure_reason": "parsed from log",
                        "failure_category": failure_category,
                        "test_cases": relevant_cases,
                    }
                )
            return report

        # Source 2b: Try to find the last running test (timeout scenario)
        last_test = find_last_running_test(log_text)
        if last_test:
            report["source"] = "log_inferred_timeout"
            report["log_truncated"] = True
            report["failed_tests"].append(
                {
                    "test_file": last_test,
                    "failure_reason": "inferred from truncated log (likely timeout)",
                    "failure_category": "timeout",
                    "test_cases": [],
                }
            )
            return report

    # Source 3: No data available — record job-level failure
    report["source"] = "job_level_only"
    report["log_truncated"] = True
    report["failed_tests"].append(
        {
            "test_file": None,
            "failure_reason": f"job-level failure (no test details available) - {job_name}",
            "failure_category": "infrastructure",
            "test_cases": [],
        }
    )

    return report


def main():
    parser = argparse.ArgumentParser(description="Extract CI test failure data")
    parser.add_argument("--job-name", required=True, help="CI job name")
    parser.add_argument("--suite", default="unknown", help="Test suite name")
    parser.add_argument("--partition-id", default=None, help="Matrix partition ID")
    parser.add_argument("--partition-size", default=None, help="Matrix partition size")
    parser.add_argument(
        "--results-json",
        default="/tmp/sglang_test_results.json",
        help="Path to structured test results JSON from ci_utils.py",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to captured log output (optional, for test-case extraction)",
    )
    parser.add_argument(
        "--output",
        default="/tmp/failure_report.json",
        help="Output path for the failure report",
    )
    args = parser.parse_args()

    # Normalize empty strings to None
    args.partition_id = args.partition_id or None
    args.partition_size = args.partition_size or None

    # Read structured test results
    test_results = parse_test_results_json(args.results_json)

    # Read log file if provided
    log_text = None
    if args.log_file and os.path.exists(args.log_file):
        try:
            with open(args.log_file, errors="ignore") as f:
                log_text = f.read()
        except Exception:
            pass

    # Build the failure report
    report = build_failure_report(
        test_results=test_results,
        log_text=log_text,
        job_name=args.job_name,
        suite=args.suite,
        partition_id=args.partition_id,
        partition_size=args.partition_size,
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    n_failed = len(report["failed_tests"])
    n_retried = len(report["retried_tests"])
    print(f"Failure report written to {output_path}")
    print(f"  Source: {report['source']}")
    print(f"  Failed tests: {n_failed}")
    print(f"  Retried tests: {n_retried}")
    if report["log_truncated"]:
        print("  Warning: Log was truncated, details may be incomplete")

    for ft in report["failed_tests"]:
        test_file = ft["test_file"] or "(unknown)"
        print(f"  - {test_file}: {ft['failure_reason']} [{ft['failure_category']}]")
        for tc in ft.get("test_cases", []):
            print(
                f"    - {tc['class_name']}.{tc['test_name']}: {tc['error_type']}: {tc['error_message'][:100]}"
            )


if __name__ == "__main__":
    main()
