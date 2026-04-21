"""Pydantic response schemas. Kept separate from DB layer so HTTP shape can
evolve independently of storage shape.
"""

from __future__ import annotations

from pydantic import BaseModel


class Metric(BaseModel):
    name: str
    value: float
    unit: str | None = None


class RunSummary(BaseModel):
    """Compact shape for lists (recent runs, config pages)."""

    id: int
    github_run_id: str
    github_run_attempt: int
    github_run_url: str
    commit_sha: str | None
    commit_short_sha: str | None
    commit_author: str | None
    pr_number: int | None
    pr_title: str | None
    trigger: str
    config_name: str
    model_prefix: str | None
    precision: str | None
    seq_len: str | None
    concurrency: int
    started_at: str
    status: str


class RunDetail(RunSummary):
    """Full run with commit message + metrics + log prefix."""

    commit_message: str | None
    commit_date: str | None
    isl: int | None
    osl: int | None
    recipe: str | None
    num_gpus: int | None
    prefill_gpus: int | None
    decode_gpus: int | None
    s3_log_prefix: str
    slurm_job_id: str | None
    ingested_at: str
    metrics: list[Metric]


class TrendPoint(BaseModel):
    """One point on a config-trend chart."""

    run_id: int
    github_run_id: str
    commit_short_sha: str | None
    commit_author: str | None
    started_at: str
    value: float


class ConfigSummary(BaseModel):
    """For home-page status cards."""

    config_name: str
    latest_run_id: int | None
    latest_started_at: str | None
    latest_status: str | None
    concurrency_levels: list[int]


class HealthStatus(BaseModel):
    status: str
    runs: int
    metrics: int
    last_ingest_at: str | None
    github_enrichment: bool
