"""GitHub REST wrapper for enriching runs with commit/PR metadata.

Degrades gracefully when `GITHUB_TOKEN` is unset — every enrichment method
returns None and logs a debug line. This lets the ingester run end-to-end
without a token; enrichment can be retried later when one is provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    short_sha: str
    message: str | None
    author_login: str | None
    date: str | None  # ISO 8601


@dataclass(frozen=True)
class PRInfo:
    number: int
    title: str | None


class GitHubClient:
    """Thin REST wrapper. Not async — caller should run in a thread pool."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, repo: str, timeout: float = 15.0) -> None:
        self._token = token
        self._repo = repo
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            timeout=timeout,
            headers=self._auth_headers(),
        )

    @property
    def enabled(self) -> bool:
        return bool(self._token)

    def _auth_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def get_workflow_run(self, run_id: str) -> dict[str, Any] | None:
        """Returns the GitHub Actions run object, or None on 404 / no token."""
        if not self.enabled:
            logger.debug("github enrichment disabled (no token); skipping run lookup")
            return None
        resp = self._client.get(f"/repos/{self._repo}/actions/runs/{run_id}")
        if resp.status_code == 404:
            return None
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            logger.warning("github rate-limited during run lookup; enrichment deferred")
            return None
        resp.raise_for_status()
        return resp.json()

    def get_commit(self, sha: str) -> CommitInfo | None:
        """Returns CommitInfo or None."""
        if not self.enabled:
            return None
        resp = self._client.get(f"/repos/{self._repo}/commits/{sha}")
        if resp.status_code == 404:
            return None
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            logger.warning("github rate-limited during commit lookup")
            return None
        resp.raise_for_status()
        data = resp.json()
        commit = data.get("commit", {})
        author = data.get("author") or {}
        return CommitInfo(
            sha=data["sha"],
            short_sha=data["sha"][:7],
            message=commit.get("message"),
            author_login=author.get("login")
            or (commit.get("author") or {}).get("name"),
            date=(commit.get("author") or {}).get("date"),
        )

    def get_pr_for_commit(self, sha: str) -> PRInfo | None:
        """Best-effort: returns first PR that merged/contains this commit."""
        if not self.enabled:
            return None
        resp = self._client.get(f"/repos/{self._repo}/commits/{sha}/pulls")
        if resp.status_code in (404, 422):
            return None
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            logger.warning("github rate-limited during PR lookup")
            return None
        resp.raise_for_status()
        items = resp.json()
        if not items:
            return None
        pr = items[0]
        return PRInfo(number=pr["number"], title=pr.get("title"))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
