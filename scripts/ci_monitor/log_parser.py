"""
Shared log parsing utilities for CI failure tracking.

Used by both:
  - scripts/ci/extract_failures.py (in-CI extraction)
  - scripts/ci_monitor/ci_failure_tracker.py (local aggregation)
  - scripts/ci_monitor/ci_failures_analysis.py (existing failure analysis)
"""

import re
from typing import Optional


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def parse_test_summary(logs: str) -> Optional[dict]:
    """
    Parse the test summary block emitted by ci_utils.py's run_unittest_files().

    Returns:
        Dict with passed/total counts and list of failed tests, or None if no summary found.
        If no summary found, attempts to find the last running test (for timeout scenarios).
    """
    logs = strip_ansi(logs)

    summary_match = re.search(r"Test Summary:\s*(\d+)/(\d+)\s*passed", logs)
    if not summary_match:
        last_test = find_last_running_test(logs)
        if last_test:
            return {
                "passed": 0,
                "total": 0,
                "failed_tests": [last_test],
                "incomplete": True,
            }
        return None

    try:
        passed = int(summary_match.group(1))
        total = int(summary_match.group(2))
    except (ValueError, TypeError):
        return None

    failed_tests = []
    failed_section_match = re.search(
        r".?\s*FAILED:\s*\n(.*?)(?:={10,}|$)", logs, re.DOTALL
    )
    if failed_section_match:
        failed_section = failed_section_match.group(1)
        # Parse "  test_file.py (exit code 1)" or "  test_file.py (timeout after 1800s)"
        for match in re.finditer(r"\s+(\S+\.py)(?:\s+\(([^)]+)\))?", failed_section):
            full_path = match.group(1)
            reason = match.group(2) or "unknown"
            test_file = full_path.split("/")[-1] if "/" in full_path else full_path
            failed_tests.append(
                {
                    "test_file": test_file,
                    "full_path": full_path,
                    "failure_reason": reason,
                }
            )

    retried_tests = []
    retried_section_match = re.search(
        r".?\s*RETRIED:\s*\n(.*?)(?:={10,}|$)", logs, re.DOTALL
    )
    if retried_section_match:
        retried_section = retried_section_match.group(1)
        for match in re.finditer(
            r"\s+(\S+\.py)\s+\((\d+)\s+attempts?,\s*(\w+)\)", retried_section
        ):
            retried_tests.append(
                {
                    "test_file": match.group(1),
                    "attempts": int(match.group(2)),
                    "result": match.group(3),
                }
            )

    return {
        "passed": passed,
        "total": total,
        "failed_tests": failed_tests,
        "retried_tests": retried_tests,
    }


def find_last_running_test(logs: str) -> Optional[dict]:
    """
    Find the last test that was running before logs cut off (for timeout/exit scenarios).
    Searches for 'Begin (N/M):' markers and 'python3 /path/to/test.py' patterns.

    Returns:
        Dict with test info if found, or None.
    """
    logs = strip_ansi(logs)
    lines = logs.split("\n")

    test_patterns = [
        r"(\S+\.py)::",  # pytest format
        r"python3?\s+(\S+\.py)",  # python3 invocation
    ]

    # Strategy 1: Find last "server_args:" and look above for the test file
    server_args_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "server_args:" in lines[i].lower() or "server_args =" in lines[i]:
            server_args_idx = i
            break

    if server_args_idx is not None:
        for j in range(1, 11):
            line_idx = server_args_idx - j
            if line_idx >= 0:
                for pattern in test_patterns:
                    match = re.search(pattern, lines[line_idx])
                    if match:
                        full_path = match.group(1)
                        test_file = (
                            full_path.split("/")[-1] if "/" in full_path else full_path
                        )
                        if test_file.endswith(".py"):
                            return {
                                "test_file": test_file,
                                "full_path": full_path,
                                "context": "last_running",
                            }

    # Strategy 2: Find the last "Begin" marker
    for i in range(len(lines) - 1, -1, -1):
        if "Begin (" in lines[i]:
            # Next non-empty line should have the python3 invocation
            for j in range(i + 1, min(i + 3, len(lines))):
                for pattern in test_patterns:
                    match = re.search(pattern, lines[j])
                    if match:
                        full_path = match.group(1)
                        test_file = (
                            full_path.split("/")[-1] if "/" in full_path else full_path
                        )
                        if test_file.endswith(".py"):
                            return {
                                "test_file": test_file,
                                "full_path": full_path,
                                "context": "last_running",
                            }

    return None


def extract_test_case_failures(log_text: str) -> list[dict]:
    """
    Parse unittest/pytest output to extract individual test case failures.

    Returns list of dicts with: class_name, test_name, error_type, error_message
    """
    test_cases = []

    # Pattern 1: unittest FAIL/ERROR blocks
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

        parts = test_class_path.rsplit(".", 1)
        class_name = parts[-1] if len(parts) > 1 else test_class_path

        test_cases.append(
            {
                "class_name": class_name,
                "test_name": test_name,
                "error_type": error_type,
                "error_message": error_message[:500],
            }
        )

    # Pattern 2: pytest FAILED lines
    pytest_pattern = re.compile(
        r"FAILED\s+(\S+\.py)::(\w+)::(\w+)(?:\s*-\s*(.+))?$", re.MULTILINE
    )
    for match in pytest_pattern.finditer(log_text):
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


def classify_failure_category(error_type: str, error_message: str) -> str:
    """Classify a failure into a high-level category."""
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
