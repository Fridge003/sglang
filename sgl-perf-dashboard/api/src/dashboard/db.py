"""SQLite schema, connection helpers, and idempotent migrations.

We use a single sqlite3 connection per request thread; APScheduler jobs get
their own. WAL mode is enabled so reader queries don't block the ingester's
writes.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                   INTEGER PRIMARY KEY,
    github_run_id        TEXT    NOT NULL,
    github_run_attempt   INTEGER NOT NULL,
    github_run_url       TEXT    NOT NULL,
    commit_sha           TEXT,
    commit_short_sha     TEXT,
    commit_message       TEXT,
    commit_author        TEXT,
    commit_date          TEXT,
    pr_number            INTEGER,
    pr_title             TEXT,
    trigger              TEXT    NOT NULL,
    config_name          TEXT    NOT NULL,
    model_prefix         TEXT,
    precision            TEXT,
    seq_len              TEXT,
    isl                  INTEGER,
    osl                  INTEGER,
    recipe               TEXT,
    concurrency          INTEGER NOT NULL,
    num_gpus             INTEGER,
    prefill_gpus         INTEGER,
    decode_gpus          INTEGER,
    started_at           TEXT    NOT NULL,
    completed_at         TEXT,
    status               TEXT    NOT NULL DEFAULT 'passed',
    s3_log_prefix        TEXT    NOT NULL,
    slurm_job_id         TEXT,
    ingested_at          TEXT    NOT NULL,
    UNIQUE (github_run_id, github_run_attempt, config_name, concurrency)
);

CREATE INDEX IF NOT EXISTS idx_runs_config_started ON runs(config_name, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_commit         ON runs(commit_sha);
CREATE INDEX IF NOT EXISTS idx_runs_pr             ON runs(pr_number) WHERE pr_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_runs_trigger_time   ON runs(trigger, started_at DESC);

CREATE TABLE IF NOT EXISTS metrics (
    id       INTEGER PRIMARY KEY,
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name     TEXT    NOT NULL,
    value    REAL    NOT NULL,
    unit     TEXT,
    UNIQUE (run_id, name)
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);

CREATE TABLE IF NOT EXISTS regressions (
    id                     INTEGER PRIMARY KEY,
    run_id                 INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    metric_name            TEXT    NOT NULL,
    severity               TEXT    NOT NULL,
    delta_percent          REAL,
    z_score                REAL,
    baseline_window_days   INTEGER,
    detected_at            TEXT    NOT NULL,
    resolved_at            TEXT,
    resolution_run_id      INTEGER REFERENCES runs(id),
    UNIQUE (run_id, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_regressions_active ON regressions(resolved_at) WHERE resolved_at IS NULL;

CREATE TABLE IF NOT EXISTS annotations (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    metric_name TEXT,
    kind        TEXT NOT NULL,
    body        TEXT NOT NULL,
    author      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_summaries (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    summary_type  TEXT    NOT NULL,
    body          TEXT    NOT NULL,
    model         TEXT,
    tokens_used   INTEGER,
    generated_at  TEXT    NOT NULL,
    UNIQUE (run_id, summary_type)
);

CREATE TABLE IF NOT EXISTS ingester_state (
    key           TEXT PRIMARY KEY,
    value         TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""

MIGRATIONS_V2 = """
CREATE TABLE IF NOT EXISTS reconciliation_state (
    workflow_run_id TEXT PRIMARY KEY,
    reconciled_at   TEXT NOT NULL
);
"""

CURRENT_SCHEMA_VERSION = 2


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Idempotent: runs SCHEMA_SQL, then records the version."""
    conn.executescript(SCHEMA_SQL)
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    installed = row[0] if row and row[0] is not None else 0

    if installed < 2:
        # Failure tracking: gh_job_url + failure_reason on runs, reconciliation_state table.
        if not _column_exists(conn, "runs", "gh_job_url"):
            conn.execute("ALTER TABLE runs ADD COLUMN gh_job_url TEXT")
        if not _column_exists(conn, "runs", "failure_reason"):
            conn.execute("ALTER TABLE runs ADD COLUMN failure_reason TEXT")
        conn.executescript(MIGRATIONS_V2)

    if installed < CURRENT_SCHEMA_VERSION:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, datetime('now'))",
            (CURRENT_SCHEMA_VERSION,),
        )
        conn.commit()
        logger.info(
            "schema migrated from v%d to v%d", installed, CURRENT_SCHEMA_VERSION
        )


def init_db(db_path: str) -> None:
    """Create the database file (if missing) and apply migrations."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        _apply_migrations(conn)


@contextmanager
def connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Short-lived connection with sane pragmas. Use for request handlers."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()
