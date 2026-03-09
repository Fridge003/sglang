#!/usr/bin/env python3
"""
Issue #17050 Auto-Updater - Generates and updates the CI failure tracking issue.

Reads the local failure database, generates markdown matching the current
issue #17050 format, and updates the issue body via `gh issue edit`.

Modes:
    --dry-run (default): Print the generated markdown without updating
    --apply: Actually update issue #17050
    --apply --ai-review: Update with AI review of fix attribution

Usage:
    # Dry run - see what would be updated
    python scripts/ci_monitor/update_tracking_issue.py

    # Actually update the issue
    python scripts/ci_monitor/update_tracking_issue.py --apply

    # With AI-assisted fix verification
    python scripts/ci_monitor/update_tracking_issue.py --apply --ai-review
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ci_monitor.ci_failure_tracker import (  # noqa: E402
    DEFAULT_DB_DIR,
    FailureDatabase,
)

REPO = "sgl-project/sglang"
TRACKING_ISSUE = 17050


def generate_issue_body(status: dict, config: dict) -> str:
    """Generate the full issue body markdown from test status data."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = []

    # Header
    sections.append(f"""# [Tracking] CI Test Failures and Fixes

This issue tracks CI test failures, flaky tests, and infrastructure issues across SGLang's CI pipeline.

**Note:** We mainly focus on scheduled CI on the `main` branch.
**Last auto-updated:** {now}

## Quick Links

- **[CI Coverage Overview](https://github.com/sgl-project/sglang/actions/workflows/ci-coverage-overview.yml)** - Check which tests are covered
- **[CI Failure Monitor](https://github.com/sgl-project/sglang/actions/workflows/ci-failure-monitor.yml)** - Automated failure analysis

---""")

    # Categorize tests
    broken_tests = []
    flaky_tests = []
    fixed_tests = []
    infra_issues = []

    for test_key, info in status.items():
        s = info["status"]
        # Infrastructure issues have no test file
        category = info.get("latest_failure_category", "")
        if category == "infrastructure" or test_key.endswith("_job_level"):
            infra_issues.append((test_key, info))
        elif s == "broken":
            broken_tests.append((test_key, info))
        elif s == "flaky":
            flaky_tests.append((test_key, info))
        elif s == "fixed":
            fixed_tests.append((test_key, info))

    # Ongoing Issues - NV CI
    sections.append("\n## Ongoing Issues\n\n### NV CI\n")

    if broken_tests or flaky_tests:
        sections.append(
            "| Date | Test | Backend | Error | Notes | CI Status | Related |"
        )
        sections.append(
            "|------|------|---------|-------|-------|-----------|---------|"
        )

        # Sort: broken first (by streak desc), then flaky (by rate desc)
        all_ongoing = []
        for test_key, info in broken_tests:
            all_ongoing.append((test_key, info, 0, -info["current_streak"]))
        for test_key, info in flaky_tests:
            all_ongoing.append((test_key, info, 1, -info["failure_rate"]))
        all_ongoing.sort(key=lambda x: (x[2], x[3]))

        for test_key, info, _, _ in all_ongoing:
            date = _format_date(info.get("last_failure", ""))
            test_name = _extract_test_name(test_key)
            backend = _format_jobs(info.get("affected_jobs", []))
            error = _format_error(info)
            notes = _format_notes(info)
            ci_status = "Enabled"
            related = _format_related(info)

            sections.append(
                f"| {date} | `{test_name}` | {backend} | {error} | {notes} | {ci_status} | {related} |"
            )
    else:
        sections.append("No ongoing test failures detected.\n")

    # Infrastructure
    if infra_issues:
        sections.append("\n### Infrastructure\n")
        sections.append("| Issue | CI Status | Related |")
        sections.append("|-------|-----------|---------|")
        for test_key, info in infra_issues:
            issue_desc = info.get("latest_error_message", test_key)[:100]
            related = _format_related(info)
            sections.append(f"| {issue_desc} | Enabled | {related} |")

    # Recently Fixed
    sections.append("""
---

<details>
<summary><h2>Recently Fixed (click to expand)</h2></summary>

""")

    if fixed_tests:
        sections.append("| Date Fixed | Test | Fix |")
        sections.append("|------------|------|-----|")

        for test_key, info in sorted(
            fixed_tests, key=lambda x: x[1].get("last_failure", ""), reverse=True
        ):
            test_name = _extract_test_name(test_key)
            date = _format_date(info.get("last_failure", ""))
            fix_pr = info.get("fix_pr")
            if fix_pr and info.get("fix_verified"):
                fix_text = f"[PR #{fix_pr}](https://github.com/{REPO}/pull/{fix_pr})"
            elif fix_pr:
                fix_text = f"[PR #{fix_pr}](https://github.com/{REPO}/pull/{fix_pr}) (unverified)"
            else:
                fix_text = "Auto-resolved (no PR attributed)"

            sections.append(f"| {date} | `{test_name}` | {fix_text} |")
    else:
        sections.append("No recently fixed tests.\n")

    sections.append("\n</details>")

    return "\n".join(sections)


def _format_date(timestamp: str) -> str:
    """Format ISO timestamp to M/D."""
    if not timestamp:
        return "-"
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return f"{dt.month}/{dt.day}"
    except (ValueError, AttributeError):
        return "-"


def _extract_test_name(test_key: str) -> str:
    """Extract human-readable test name from key like 'test_foo.py::TestBar.test_baz'."""
    # Remove .py extension for display
    if "::" in test_key:
        parts = test_key.split("::")
        return parts[0]  # Just the file name
    return test_key


def _format_jobs(jobs: list[str]) -> str:
    """Format job names for the Backend column."""
    if not jobs:
        return "-"
    # Shorten common prefixes
    formatted = []
    for job in jobs[:2]:
        job = job.replace("stage-b-test-", "Stage-B ")
        job = job.replace("stage-c-test-", "Stage-C ")
        job = job.replace("stage-a-test-", "Stage-A ")
        formatted.append(job)
    return ", ".join(formatted)


def _format_error(info: dict) -> str:
    """Format error info for display."""
    error_type = info.get("latest_error_type", "")
    error_msg = info.get("latest_error_message", "")
    category = info.get("latest_failure_category", "")

    if error_type and error_msg:
        return f"`{error_type}`: {error_msg[:60]}"
    elif error_msg:
        return error_msg[:80]
    elif category:
        return category
    return "-"


def _format_notes(info: dict) -> str:
    """Format notes column."""
    parts = []
    status = info["status"]
    if status == "broken":
        parts.append(f"Failing {info['current_streak']}x consecutively")
    elif status == "flaky":
        parts.append(f"Flaky ({info['failure_rate']:.0%} failure rate)")

    runners = info.get("affected_runners", [])
    if runners and len(runners) == 1:
        parts.append(f"Runner: {runners[0]}")

    return "; ".join(parts) if parts else "-"


def _format_related(info: dict) -> str:
    """Format the Related column with links."""
    url = info.get("latest_job_url", "")
    if url:
        return f"[Link]({url})"
    return "-"


def find_fix_candidates(test_key: str, info: dict, github_token: str) -> list[dict]:
    """
    Find candidate PRs that may have fixed a test.

    Looks at PRs merged between the last failure and the first subsequent success.
    """
    last_failure_sha = info.get("latest_head_sha", "")
    if not last_failure_sha:
        return []

    # Use git log to find commits after the failure
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--oneline",
                "--format=%H %s",
                f"{last_failure_sha}..HEAD",
                "--",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30,
        )
        if result.returncode != 0:
            return []

        # Extract test file name for searching
        test_file = test_key.split("::")[0] if "::" in test_key else test_key

        candidates = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            sha, message = line.split(" ", 1)
            # Check if commit message references the test
            if test_file.replace(".py", "") in message.lower() or test_file in message:
                candidates.append({"sha": sha, "message": message})

        return candidates[:5]  # Limit to 5 candidates

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def ai_review_fix(test_key: str, info: dict, candidate_pr: dict) -> bool:
    """
    Use AI to review whether a candidate PR actually fixed the test.

    This is a placeholder that prints a review prompt. In production,
    this would call an LLM API to verify the fix attribution.

    Returns True if the fix is verified, False otherwise.
    """
    test_file = test_key.split("::")[0] if "::" in test_key else test_key
    error = info.get("latest_error_message", "unknown")
    pr_message = candidate_pr.get("message", "")

    print("\n--- AI Review Needed ---")
    print(f"Test: {test_file}")
    print(f"Error: {error}")
    print(f"Candidate fix: {pr_message}")
    print("Question: Did this PR fix the test failure?")
    print("[Automated AI review not yet implemented - marking as unverified]")
    print("------------------------\n")

    # TODO: Integrate with Claude API for automated review
    # For now, return False (unverified) to be safe
    return False


def backup_current_issue(issue_number: int, backup_dir: Path):
    """Backup current issue body before updating."""
    backup_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["gh", "issue", "view", str(issue_number), "--json", "body"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"issue_{issue_number}_{timestamp}.md"
            with open(backup_path, "w") as f:
                f.write(data.get("body", ""))
            print(f"Backed up current issue body to {backup_path}")

            # Keep only last 10 backups
            backups = sorted(
                backup_dir.glob(f"issue_{issue_number}_*.md"), reverse=True
            )
            for old in backups[10:]:
                old.unlink()

    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"Warning: Could not backup issue: {e}")


def update_issue(issue_number: int, body: str):
    """Update the issue body via gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "issue", "edit", str(issue_number), "--body", body],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            print(f"Successfully updated issue #{issue_number}")
        else:
            print(f"Error updating issue: {result.stderr}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Error: gh command timed out")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Update issue #17050 with CI failure tracking data"
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DB_DIR,
        help=f"Database directory (default: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update the issue (default is dry-run)",
    )
    parser.add_argument(
        "--ai-review",
        action="store_true",
        help="Enable AI review for fix attribution",
    )
    parser.add_argument(
        "--issue",
        type=int,
        default=TRACKING_ISSUE,
        help=f"Issue number to update (default: {TRACKING_ISSUE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write generated markdown to file instead of stdout",
    )
    args = parser.parse_args()

    db = FailureDatabase(args.db_dir)
    status = db.get_test_status()
    config = db.get_config()

    if not status:
        print("No test status data. Run 'ci_failure_tracker.py sync' first.")
        sys.exit(1)

    # AI review for fix attribution
    if args.ai_review:
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        for test_key, info in status.items():
            if info["status"] == "fixed" and not info.get("fix_verified"):
                candidates = find_fix_candidates(test_key, info, github_token)
                if candidates:
                    verified = ai_review_fix(test_key, info, candidates[0])
                    if verified:
                        info["fix_verified"] = True
                        info["fix_pr"] = candidates[0].get("pr_number")
        db.update_test_status(status)

    # Generate the issue body
    body = generate_issue_body(status, config)

    if args.output:
        with open(args.output, "w") as f:
            f.write(body)
        print(f"Written to {args.output}")

    if args.apply:
        backup_dir = args.db_dir / "issue_backups"
        backup_current_issue(args.issue, backup_dir)
        update_issue(args.issue, body)
    else:
        print("=== DRY RUN - Generated issue body ===\n")
        print(body)
        print("\n=== End of dry run ===")
        print("\nTo apply, run with --apply")


if __name__ == "__main__":
    main()
