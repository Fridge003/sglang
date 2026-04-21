"""FastAPI entry point. Exposes /api/* routes and runs the APScheduler-driven
ingester alongside the HTTP server in the same process.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from dashboard import ingester
from dashboard.config import settings
from dashboard.db import connect, init_db
from dashboard.models import (
    ConfigSummary,
    HealthStatus,
    Metric,
    RunDetail,
    RunSummary,
    TrendPoint,
)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_scheduler: BackgroundScheduler | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db(settings.db_path)
    logger.info("db initialized at %s", settings.db_path)

    global _scheduler
    _scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
    _scheduler.add_job(
        _safe_ingest,
        "interval",
        seconds=settings.ingester_interval_seconds,
        id="ingest",
        # Fire immediately on startup so first deploy doesn't wait 5 min.
        next_run_time=None,
    )
    _scheduler.start()
    logger.info(
        "scheduler started; ingest cadence=%ds, github enrichment=%s",
        settings.ingester_interval_seconds,
        settings.github_enrichment_enabled,
    )

    # Kick off an immediate ingest in the background (non-blocking).
    _scheduler.add_job(_safe_ingest, id="ingest-bootstrap")

    try:
        yield
    finally:
        if _scheduler:
            _scheduler.shutdown(wait=False)


def _safe_ingest() -> None:
    try:
        ingester.run_once()
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingester run failed: %s", exc)


app = FastAPI(
    title="sglang Perf Dashboard",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten when auth lands
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


def _row_to_summary(row: sqlite3.Row) -> RunSummary:
    return RunSummary(
        id=row["id"],
        github_run_id=row["github_run_id"],
        github_run_attempt=row["github_run_attempt"],
        github_run_url=row["github_run_url"],
        commit_sha=row["commit_sha"],
        commit_short_sha=row["commit_short_sha"],
        commit_author=row["commit_author"],
        pr_number=row["pr_number"],
        pr_title=row["pr_title"],
        trigger=row["trigger"],
        config_name=row["config_name"],
        model_prefix=row["model_prefix"],
        precision=row["precision"],
        seq_len=row["seq_len"],
        concurrency=row["concurrency"],
        started_at=row["started_at"],
        status=row["status"],
    )


def _row_to_detail(row: sqlite3.Row, metrics: list[Metric]) -> RunDetail:
    return RunDetail(
        id=row["id"],
        github_run_id=row["github_run_id"],
        github_run_attempt=row["github_run_attempt"],
        github_run_url=row["github_run_url"],
        commit_sha=row["commit_sha"],
        commit_short_sha=row["commit_short_sha"],
        commit_author=row["commit_author"],
        commit_message=row["commit_message"],
        commit_date=row["commit_date"],
        pr_number=row["pr_number"],
        pr_title=row["pr_title"],
        trigger=row["trigger"],
        config_name=row["config_name"],
        model_prefix=row["model_prefix"],
        precision=row["precision"],
        seq_len=row["seq_len"],
        isl=row["isl"],
        osl=row["osl"],
        recipe=row["recipe"],
        concurrency=row["concurrency"],
        num_gpus=row["num_gpus"],
        prefill_gpus=row["prefill_gpus"],
        decode_gpus=row["decode_gpus"],
        started_at=row["started_at"],
        status=row["status"],
        s3_log_prefix=row["s3_log_prefix"],
        slurm_job_id=row["slurm_job_id"],
        ingested_at=row["ingested_at"],
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthStatus)
def health() -> HealthStatus:
    with connect(settings.db_path) as conn:
        runs_count = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
        metrics_count = conn.execute("SELECT COUNT(*) AS c FROM metrics").fetchone()[
            "c"
        ]
        cursor_row = conn.execute(
            "SELECT updated_at FROM ingester_state WHERE key = 's3_cursor'"
        ).fetchone()
    return HealthStatus(
        status="ok",
        runs=runs_count,
        metrics=metrics_count,
        last_ingest_at=cursor_row["updated_at"] if cursor_row else None,
        github_enrichment=settings.github_enrichment_enabled,
    )


@app.get("/api/runs", response_model=list[RunSummary])
def list_runs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    config: str | None = None,
    trigger: str | None = None,
    status: str | None = None,
) -> list[RunSummary]:
    clauses: list[str] = []
    params: list[Any] = []
    if config:
        clauses.append("config_name = ?")
        params.append(config)
    if trigger:
        clauses.append("trigger = ?")
        params.append(trigger)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT * FROM runs
        {where}
        ORDER BY started_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    with connect(settings.db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_summary(r) for r in rows]


@app.get("/api/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: int) -> RunDetail:
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        metric_rows = conn.execute(
            "SELECT name, value, unit FROM metrics WHERE run_id = ? ORDER BY name",
            (run_id,),
        ).fetchall()
    metrics = [
        Metric(name=m["name"], value=m["value"], unit=m["unit"]) for m in metric_rows
    ]
    return _row_to_detail(row, metrics)


@app.get("/api/configs", response_model=list[ConfigSummary])
def list_configs() -> list[ConfigSummary]:
    with connect(settings.db_path) as conn:
        rows = conn.execute("""
            SELECT
                config_name,
                MAX(started_at) AS latest_started_at,
                GROUP_CONCAT(DISTINCT concurrency) AS concurrency_csv
            FROM runs
            GROUP BY config_name
            ORDER BY config_name
            """).fetchall()
        out = []
        for r in rows:
            latest = conn.execute(
                """
                SELECT id, status FROM runs
                WHERE config_name = ? AND started_at = ?
                LIMIT 1
                """,
                (r["config_name"], r["latest_started_at"]),
            ).fetchone()
            concs = sorted(int(c) for c in (r["concurrency_csv"] or "").split(",") if c)
            out.append(
                ConfigSummary(
                    config_name=r["config_name"],
                    latest_run_id=latest["id"] if latest else None,
                    latest_started_at=r["latest_started_at"],
                    latest_status=latest["status"] if latest else None,
                    concurrency_levels=concs,
                )
            )
    return out


@app.get("/api/configs/{config_name}/trend", response_model=list[TrendPoint])
def config_trend(
    config_name: str,
    metric: str = Query(
        ..., description="Metric name (verbatim, e.g. 'total_token_throughput')"
    ),
    concurrency: int = Query(..., ge=1),
    window_days: int = Query(default=30, ge=1, le=365),
) -> list[TrendPoint]:
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.id AS run_id, r.github_run_id, r.commit_short_sha, r.commit_author,
                   r.started_at, m.value
            FROM runs r
            JOIN metrics m ON m.run_id = r.id
            WHERE r.config_name = ?
              AND r.concurrency = ?
              AND m.name = ?
              AND r.started_at > datetime('now', ?)
              AND r.status = 'passed'
            ORDER BY r.started_at
            """,
            (config_name, concurrency, metric, f"-{window_days} day"),
        ).fetchall()
    return [
        TrendPoint(
            run_id=r["run_id"],
            github_run_id=r["github_run_id"],
            commit_short_sha=r["commit_short_sha"],
            commit_author=r["commit_author"],
            started_at=r["started_at"],
            value=r["value"],
        )
        for r in rows
    ]


@app.post("/api/admin/ingest")
def trigger_ingest() -> dict[str, Any]:
    """Manual trigger — useful during M1 testing. Replaces scheduled waiting."""
    stats = ingester.run_once()
    return {"ok": True, "stats": stats}
