#!/usr/bin/env python3
"""
CI Failure Tracker - Local aggregator and database for CI test failures.

Fetches failure data from GitHub Actions (artifacts + job logs), stores in a
local JSONL database, and tracks failure patterns (broken, flaky, fixed, new).

Database location: ~/.sglang/ci_failures/

Usage:
    # Sync recent failures from GitHub
    python scripts/ci_monitor/ci_failure_tracker.py sync --hours 48

    # Show current status of all tracked tests
    python scripts/ci_monitor/ci_failure_tracker.py status

    # Show details for a specific test
    python scripts/ci_monitor/ci_failure_tracker.py status --test test_autoround.py

    # Export current status as JSON (for issue updater)
    python scripts/ci_monitor/ci_failure_tracker.py export --output /tmp/ci_status.json
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ci_monitor.log_parser import (  # noqa: E402
    classify_failure_category,
    extract_test_case_failures,
    parse_test_summary,
)

# Default database directory
DEFAULT_DB_DIR = Path.home() / ".sglang" / "ci_failures"

# GitHub repo
REPO = "sgl-project/sglang"
GITHUB_API = "https://api.github.com"


class FailureDatabase:
    """Manages the local JSONL failure database."""

    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.failures_path = db_dir / "failures.jsonl"
        self.status_path = db_dir / "test_status.json"
        self.config_path = db_dir / "config.json"
        self.run_cache_path = db_dir / "run_cache.json"

        self._ensure_config()

    def _ensure_config(self):
        if not self.config_path.exists():
            config = {
                "repo": REPO,
                "tracking_issue": 17050,
                "focus_events": ["schedule"],
                "focus_branch": "main",
                "workflows": ["pr-test.yml"],
                "broken_threshold": 2,
                "flaky_rate_threshold": 0.10,
                "fixed_consecutive_passes": 3,
                "last_processed_run_id": None,
                "db_version": "1.0",
            }
            self._write_json(self.config_path, config)

    def get_config(self) -> dict:
        return self._read_json(self.config_path)

    def update_config(self, updates: dict):
        config = self.get_config()
        config.update(updates)
        self._write_json(self.config_path, config)

    def get_run_cache(self) -> dict:
        if self.run_cache_path.exists():
            return self._read_json(self.run_cache_path)
        return {"processed_run_ids": []}

    def update_run_cache(self, run_ids: list[int]):
        cache = self.get_run_cache()
        existing = set(cache.get("processed_run_ids", []))
        existing.update(run_ids)
        # Keep last 500 run IDs
        sorted_ids = sorted(existing, reverse=True)[:500]
        cache["processed_run_ids"] = sorted_ids
        self._write_json(self.run_cache_path, cache)

    def append_failures(self, records: list[dict]):
        """Append failure records to JSONL database."""
        with open(self.failures_path, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def read_all_failures(self) -> list[dict]:
        """Read all failure records."""
        if not self.failures_path.exists():
            return []
        records = []
        with open(self.failures_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_test_status(self) -> dict:
        """Read current test status."""
        if self.status_path.exists():
            return self._read_json(self.status_path)
        return {}

    def update_test_status(self, status: dict):
        self._write_json(self.status_path, status)

    def _read_json(self, path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    def _write_json(self, path: Path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class GitHubClient:
    """Minimal GitHub API client for fetching CI data."""

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            }
        )

    def get_recent_runs(
        self,
        workflow: str,
        branch: str = "main",
        event: str = "schedule",
        limit: int = 50,
    ) -> list[dict]:
        """Fetch recent workflow runs."""
        url = f"{GITHUB_API}/repos/{REPO}/actions/workflows/{workflow}/runs"
        params = {
            "per_page": min(limit, 100),
            "status": "completed",
            "branch": branch,
            "event": event,
        }
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("workflow_runs", [])[:limit]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching runs: {e}")
            return []

    def get_jobs_for_run(self, run_id: int) -> list[dict]:
        """Get all jobs for a workflow run (handles pagination)."""
        all_jobs = []
        url = f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}/jobs"
        params = {"per_page": 100}

        while url:
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                all_jobs.extend(data.get("jobs", []))

                # Handle pagination
                url = None
                link = resp.headers.get("Link", "")
                if link:
                    for part in link.split(", "):
                        if 'rel="next"' in part:
                            url = part.split(";")[0].strip("<>")
                            break
                params = {}
            except requests.exceptions.RequestException:
                break

        return all_jobs

    def get_job_logs(self, job_id: int) -> str:
        """Fetch logs for a specific job."""
        try:
            url = f"{GITHUB_API}/repos/{REPO}/actions/jobs/{job_id}/logs"
            resp = self.session.get(url, timeout=60, allow_redirects=True)
            if resp.status_code == 200:
                return resp.text
            return ""
        except requests.exceptions.RequestException:
            return ""

    def get_run_artifacts(self, run_id: int) -> list[dict]:
        """List artifacts for a workflow run."""
        try:
            url = f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}/artifacts"
            resp = self.session.get(url, params={"per_page": 100}, timeout=30)
            resp.raise_for_status()
            return resp.json().get("artifacts", [])
        except requests.exceptions.RequestException:
            return []

    def download_artifact(self, artifact_id: int) -> Optional[dict]:
        """Download and parse a failure report artifact (ZIP containing JSON)."""
        try:
            url = f"{GITHUB_API}/repos/{REPO}/actions/artifacts/{artifact_id}/zip"
            resp = self.session.get(url, timeout=60, allow_redirects=True)
            if resp.status_code != 200:
                return None

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in zf.namelist():
                    if name.endswith(".json"):
                        with zf.open(name) as f:
                            return json.load(f)
            return None
        except Exception:
            return None

    def get_merged_prs(self, since: str, until: str) -> list[dict]:
        """Get PRs merged in a time range."""
        try:
            # Use search API to find merged PRs
            query = f"repo:{REPO} is:pr is:merged merged:{since}..{until}"
            url = f"{GITHUB_API}/search/issues"
            resp = self.session.get(
                url, params={"q": query, "per_page": 100}, timeout=30
            )
            resp.raise_for_status()
            return resp.json().get("items", [])
        except requests.exceptions.RequestException:
            return []


def sync_failures(db: FailureDatabase, github: GitHubClient, hours: int = 48):
    """Sync recent failures from GitHub to the local database."""
    config = db.get_config()
    cache = db.get_run_cache()
    processed = set(cache.get("processed_run_ids", []))

    new_records = []
    new_run_ids = []

    for workflow in config.get("workflows", ["pr-test.yml"]):
        for event in config.get("focus_events", ["schedule"]):
            print(f"Fetching {event} runs for {workflow}...")
            runs = github.get_recent_runs(
                workflow=workflow,
                branch=config.get("focus_branch", "main"),
                event=event,
                limit=50,
            )

            for run in runs:
                run_id = run["id"]
                if run_id in processed:
                    continue

                run_number = run.get("run_number", 0)
                conclusion = run.get("conclusion", "")
                head_sha = run.get("head_sha", "")
                created_at = run.get("created_at", "")

                print(f"  Processing run #{run_number} ({conclusion})...")

                # Try to get failure artifacts first
                artifacts = github.get_run_artifacts(run_id)
                failure_artifacts = [
                    a for a in artifacts if a["name"].startswith("failure-report-")
                ]

                if failure_artifacts:
                    for artifact in failure_artifacts:
                        print(f"    Downloading artifact: {artifact['name']}")
                        report = github.download_artifact(artifact["id"])
                        if report:
                            records = _artifact_to_records(report, run, event)
                            new_records.extend(records)
                        time.sleep(0.2)  # Rate limiting
                else:
                    # Fallback: parse job logs directly
                    if conclusion == "failure":
                        jobs = github.get_jobs_for_run(run_id)
                        failed_jobs = [
                            j for j in jobs if j.get("conclusion") == "failure"
                        ]

                        for job in failed_jobs:
                            job_name = job.get("name", "unknown")
                            job_id = job.get("id")

                            # Skip administrative jobs
                            if any(
                                skip in job_name.lower()
                                for skip in [
                                    "check-changes",
                                    "pr-test-finish",
                                    "call-gate",
                                    "wait-for-",
                                    "sgl-kernel-build",
                                ]
                            ):
                                continue

                            print(f"    Parsing logs for: {job_name}")
                            logs = github.get_job_logs(job_id)
                            if logs:
                                records = _logs_to_records(logs, job, run, event)
                                new_records.extend(records)
                            time.sleep(0.3)  # Rate limiting for log fetches

                    # Also record successful runs for streak tracking
                    elif conclusion == "success":
                        new_records.append(
                            {
                                "id": f"run_{run_id}_success",
                                "timestamp": created_at,
                                "run_id": run_id,
                                "run_number": run_number,
                                "event": event,
                                "branch": config.get("focus_branch", "main"),
                                "head_sha": head_sha,
                                "conclusion": "success",
                                "test_file": None,
                                "test_case": None,
                            }
                        )

                new_run_ids.append(run_id)
                time.sleep(0.1)

    if new_records:
        db.append_failures(new_records)
        print(f"\nAdded {len(new_records)} records to database")

    if new_run_ids:
        db.update_run_cache(new_run_ids)
        print(f"Processed {len(new_run_ids)} new runs")

    # Recompute test statuses
    _recompute_statuses(db)


def _artifact_to_records(report: dict, run: dict, event: str) -> list[dict]:
    """Convert a failure report artifact to database records."""
    records = []
    run_id = run["id"]
    run_number = run.get("run_number", 0)
    head_sha = run.get("head_sha", "")
    created_at = run.get("created_at", "")

    for failed in report.get("failed_tests", []):
        test_file = failed.get("test_file")
        failure_reason = failed.get("failure_reason", "unknown")
        failure_category = failed.get("failure_category", "other")

        # Create a record per test case if available, otherwise per test file
        test_cases = failed.get("test_cases", [])
        if test_cases:
            for tc in test_cases:
                record_id = f"run_{run_id}_{report.get('job_name', 'unknown')}_{test_file}_{tc['class_name']}.{tc['test_name']}"
                records.append(
                    {
                        "id": record_id,
                        "timestamp": report.get("timestamp", created_at),
                        "run_id": run_id,
                        "run_number": run_number,
                        "job_name": report.get("job_name", "unknown"),
                        "job_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
                        "event": event,
                        "branch": report.get("branch", "main"),
                        "head_sha": report.get("head_sha", head_sha),
                        "pr_number": report.get("pr_number"),
                        "runner_name": report.get("runner_name", "unknown"),
                        "test_file": test_file,
                        "test_case": f"{tc['class_name']}.{tc['test_name']}",
                        "error_type": tc.get("error_type", "Unknown"),
                        "error_message": tc.get("error_message", ""),
                        "failure_category": classify_failure_category(
                            tc.get("error_type", ""),
                            tc.get("error_message", ""),
                        ),
                        "source": "artifact",
                        "conclusion": "failure",
                    }
                )
        elif test_file:
            record_id = f"run_{run_id}_{report.get('job_name', 'unknown')}_{test_file}"
            records.append(
                {
                    "id": record_id,
                    "timestamp": report.get("timestamp", created_at),
                    "run_id": run_id,
                    "run_number": run_number,
                    "job_name": report.get("job_name", "unknown"),
                    "job_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
                    "event": event,
                    "branch": report.get("branch", "main"),
                    "head_sha": report.get("head_sha", head_sha),
                    "pr_number": report.get("pr_number"),
                    "runner_name": report.get("runner_name", "unknown"),
                    "test_file": test_file,
                    "test_case": None,
                    "error_type": None,
                    "error_message": failure_reason,
                    "failure_category": failure_category,
                    "source": "artifact",
                    "conclusion": "failure",
                }
            )

    return records


def _logs_to_records(logs: str, job: dict, run: dict, event: str) -> list[dict]:
    """Convert parsed job logs to database records."""
    records = []
    run_id = run["id"]
    run_number = run.get("run_number", 0)
    head_sha = run.get("head_sha", "")
    created_at = run.get("created_at", "")
    job_name = job.get("name", "unknown")
    job_id = job.get("id")
    runner_name = job.get("runner_name", "unknown")

    job_url = f"https://github.com/{REPO}/actions/runs/{run_id}/job/{job_id}"

    summary = parse_test_summary(logs)
    if summary:
        for failed_test in summary.get("failed_tests", []):
            test_file = failed_test.get("test_file", "unknown")
            failure_reason = failed_test.get("failure_reason", "unknown")
            context = failed_test.get("context", "")

            failure_category = "timeout" if context == "last_running" else "other"
            if "timeout" in failure_reason:
                failure_category = "timeout"

            # Try to extract test cases
            test_cases = extract_test_case_failures(logs)
            file_base = test_file.replace(".py", "").replace("test_", "")
            relevant_cases = [
                tc for tc in test_cases if file_base in tc.get("class_name", "").lower()
            ]

            if relevant_cases:
                for tc in relevant_cases:
                    record_id = f"run_{run_id}_{job_name}_{test_file}_{tc['class_name']}.{tc['test_name']}"
                    records.append(
                        {
                            "id": record_id,
                            "timestamp": created_at,
                            "run_id": run_id,
                            "run_number": run_number,
                            "job_name": job_name,
                            "job_id": job_id,
                            "job_url": job_url,
                            "event": event,
                            "branch": "main",
                            "head_sha": head_sha,
                            "pr_number": None,
                            "runner_name": runner_name,
                            "test_file": test_file,
                            "test_case": f"{tc['class_name']}.{tc['test_name']}",
                            "error_type": tc.get("error_type"),
                            "error_message": tc.get("error_message", ""),
                            "failure_category": classify_failure_category(
                                tc.get("error_type", ""),
                                tc.get("error_message", ""),
                            ),
                            "source": "log_parsing",
                            "conclusion": "failure",
                        }
                    )
            else:
                record_id = f"run_{run_id}_{job_name}_{test_file}"
                records.append(
                    {
                        "id": record_id,
                        "timestamp": created_at,
                        "run_id": run_id,
                        "run_number": run_number,
                        "job_name": job_name,
                        "job_id": job_id,
                        "job_url": job_url,
                        "event": event,
                        "branch": "main",
                        "head_sha": head_sha,
                        "pr_number": None,
                        "runner_name": runner_name,
                        "test_file": test_file,
                        "test_case": None,
                        "error_type": None,
                        "error_message": failure_reason,
                        "failure_category": failure_category,
                        "source": "log_parsing",
                        "conclusion": "failure",
                    }
                )
    else:
        # No test summary — record as job-level failure
        records.append(
            {
                "id": f"run_{run_id}_{job_name}_job_level",
                "timestamp": created_at,
                "run_id": run_id,
                "run_number": run_number,
                "job_name": job_name,
                "job_id": job_id,
                "job_url": job_url,
                "event": event,
                "branch": "main",
                "head_sha": head_sha,
                "pr_number": None,
                "runner_name": runner_name,
                "test_file": None,
                "test_case": None,
                "error_type": None,
                "error_message": "job-level failure (no test summary)",
                "failure_category": "infrastructure",
                "source": "job_level",
                "conclusion": "failure",
            }
        )

    return records


def _recompute_statuses(db: FailureDatabase):
    """Recompute test_status.json from the full failure history."""
    config = db.get_config()
    all_records = db.read_all_failures()

    if not all_records:
        return

    # Group records by test identifier (test_file, or test_file::test_case)
    test_records: dict[str, list[dict]] = defaultdict(list)
    success_run_ids: set[int] = set()

    for record in all_records:
        if record.get("conclusion") == "success":
            success_run_ids.add(record.get("run_id"))
            continue

        test_file = record.get("test_file")
        if not test_file:
            continue

        test_case = record.get("test_case")
        key = f"{test_file}::{test_case}" if test_case else test_file
        test_records[key].append(record)

    # Get all unique run IDs for ordering
    all_run_ids = set()
    for records in test_records.values():
        for r in records:
            all_run_ids.add(r.get("run_id"))
    all_run_ids.update(success_run_ids)
    sorted_run_ids = sorted(all_run_ids)

    # Compute status for each test
    broken_threshold = config.get("broken_threshold", 2)
    flaky_threshold = config.get("flaky_rate_threshold", 0.10)
    fixed_passes = config.get("fixed_consecutive_passes", 3)

    status = {}
    for test_key, records in test_records.items():
        # Sort records by run_id (chronological)
        records.sort(key=lambda r: r.get("run_id", 0))

        failure_run_ids = {r.get("run_id") for r in records}
        total_runs = len(sorted_run_ids)
        total_failures = len(failure_run_ids)

        # Compute current streak (consecutive failures from most recent)
        current_streak = 0
        for run_id in reversed(sorted_run_ids):
            if run_id in failure_run_ids:
                current_streak += 1
            elif run_id in success_run_ids:
                break
            # Skip cancelled/unknown runs

        # Compute max streak
        max_streak = 0
        streak = 0
        for run_id in sorted_run_ids:
            if run_id in failure_run_ids:
                streak += 1
                max_streak = max(max_streak, streak)
            elif run_id in success_run_ids:
                streak = 0

        # Consecutive passes since last failure
        consecutive_passes = 0
        for run_id in reversed(sorted_run_ids):
            if run_id in success_run_ids and run_id not in failure_run_ids:
                consecutive_passes += 1
            elif run_id in failure_run_ids:
                break

        # Classify status
        failure_rate = total_failures / max(total_runs, 1)
        if current_streak >= broken_threshold:
            test_status = "broken"
        elif consecutive_passes >= fixed_passes and total_failures > 0:
            test_status = "fixed"
        elif failure_rate >= flaky_threshold and current_streak < broken_threshold:
            test_status = "flaky"
        elif total_runs < 3:
            test_status = "new"
        else:
            test_status = "occasional"

        # Get metadata from records
        affected_jobs = list(
            {r.get("job_name", "") for r in records if r.get("job_name")}
        )
        affected_runners = list(
            {r.get("runner_name", "") for r in records if r.get("runner_name")}
        )
        timestamps = [r.get("timestamp", "") for r in records]
        first_seen = min(timestamps) if timestamps else ""
        last_failure = max(timestamps) if timestamps else ""

        # Most recent error info
        latest = records[-1] if records else {}

        status[test_key] = {
            "status": test_status,
            "current_streak": current_streak,
            "max_streak": max_streak,
            "consecutive_passes": consecutive_passes,
            "total_failures": total_failures,
            "total_runs": total_runs,
            "failure_rate": round(failure_rate, 3),
            "first_seen": first_seen,
            "last_failure": last_failure,
            "affected_jobs": affected_jobs[:5],  # Limit to 5
            "affected_runners": affected_runners[:5],
            "latest_error_type": latest.get("error_type"),
            "latest_error_message": latest.get("error_message", "")[:200],
            "latest_failure_category": latest.get("failure_category"),
            "latest_job_url": latest.get("job_url"),
            "latest_run_id": latest.get("run_id"),
            "latest_head_sha": latest.get("head_sha"),
            "fix_pr": None,
            "fix_verified": False,
            "notes": "",
        }

    db.update_test_status(status)
    print(f"Updated status for {len(status)} tracked tests")

    # Print summary
    by_status = defaultdict(int)
    for s in status.values():
        by_status[s["status"]] += 1
    for st, count in sorted(by_status.items()):
        print(f"  {st}: {count}")


def show_status(db: FailureDatabase, test_filter: Optional[str] = None):
    """Display current test failure status."""
    status = db.get_test_status()

    if not status:
        print("No tracked tests. Run 'sync' first.")
        return

    if test_filter:
        filtered = {k: v for k, v in status.items() if test_filter in k}
        if not filtered:
            print(f"No tests matching '{test_filter}'")
            return
        status = filtered

    # Group by status
    groups = defaultdict(list)
    for test_key, info in status.items():
        groups[info["status"]].append((test_key, info))

    status_order = ["broken", "flaky", "new", "occasional", "fixed"]
    status_emoji = {
        "broken": "X",
        "flaky": "~",
        "new": "?",
        "occasional": ".",
        "fixed": "V",
    }

    for s in status_order:
        tests = groups.get(s, [])
        if not tests:
            continue

        print(f"\n[{status_emoji.get(s, ' ')}] {s.upper()} ({len(tests)})")
        print("-" * 80)

        for test_key, info in sorted(tests, key=lambda x: -x[1]["current_streak"]):
            streak = info["current_streak"]
            rate = info["failure_rate"]
            category = info.get("latest_failure_category", "")
            jobs = ", ".join(info.get("affected_jobs", [])[:2])
            error_msg = info.get("latest_error_message", "")[:80]

            print(f"  {test_key}")
            print(f"    streak={streak}, rate={rate:.1%}, category={category}")
            if jobs:
                print(f"    jobs: {jobs}")
            if error_msg:
                print(f"    error: {error_msg}")
            if info.get("latest_job_url"):
                print(f"    url: {info['latest_job_url']}")
            print()


def export_status(db: FailureDatabase, output: str):
    """Export current status as JSON for the issue updater."""
    status = db.get_test_status()
    config = db.get_config()

    export = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "test_statuses": status,
        "summary": {
            "broken": len([v for v in status.values() if v["status"] == "broken"]),
            "flaky": len([v for v in status.values() if v["status"] == "flaky"]),
            "new": len([v for v in status.values() if v["status"] == "new"]),
            "fixed": len([v for v in status.values() if v["status"] == "fixed"]),
            "occasional": len(
                [v for v in status.values() if v["status"] == "occasional"]
            ),
        },
    }

    with open(output, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Exported status to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="CI Failure Tracker - Local aggregator for CI test failures"
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DB_DIR,
        help=f"Database directory (default: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
        help="GitHub token (default: $GITHUB_TOKEN or $GH_TOKEN)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync recent failures from GitHub")
    sync_parser.add_argument(
        "--hours", type=int, default=48, help="Hours of history to sync"
    )

    # status command
    status_parser = subparsers.add_parser("status", help="Show current test status")
    status_parser.add_argument("--test", help="Filter by test name")

    # export command
    export_parser = subparsers.add_parser("export", help="Export status as JSON")
    export_parser.add_argument(
        "--output", default="/tmp/ci_status.json", help="Output path"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    db = FailureDatabase(args.db_dir)

    if args.command == "sync":
        if not args.token:
            print(
                "Error: GitHub token required. Set GITHUB_TOKEN env var or use --token"
            )
            sys.exit(1)
        github = GitHubClient(args.token)
        sync_failures(db, github, hours=args.hours)

    elif args.command == "status":
        show_status(db, test_filter=args.test)

    elif args.command == "export":
        export_status(db, args.output)


if __name__ == "__main__":
    main()
