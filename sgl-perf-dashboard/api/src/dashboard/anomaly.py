"""Rolling-window anomaly detection for benchmark metrics.

The ingester calls `detect_for_run()` after every fresh insert. We walk every
metric stored for that run and compare it to the rolling window of historical
values for the same (config, concurrency, metric) triple. Anything more than
`Z_THRESHOLD` median-absolute-deviations from the rolling median becomes a
regression row.

We use MAD rather than stddev because:
  - MAD is robust to outliers (one bad day won't inflate the baseline)
  - Scaled MAD ≈ stddev for normal distributions, so z-scores remain
    intuitively readable (|z| > 2.5 ≈ ~1% false-positive rate).

If a previously-flagged regression is followed by a passing run at the same
coordinate, we auto-resolve it.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

logger = logging.getLogger(__name__)

# MAD-based z-score above which we flag. 2.5σ ≈ ~1% baseline FP rate.
Z_THRESHOLD = 2.5
# Window of "recent history" we compare against.
DEFAULT_WINDOW_DAYS = 30
# Minimum sample count before detection runs — too few points produces noise.
MIN_SAMPLES = 5
# MAD→σ scaling constant for a normal distribution.
MAD_TO_SIGMA = 1.4826

# Some metrics (throughput) are "higher is better" → low outliers are bad.
# Others (latency) are "lower is better" → high outliers are bad.
# We flag any direction, and let severity account for the sign.
HIGHER_IS_BETTER_PREFIXES = (
    "throughput",
    "total_token_throughput",
    "request_throughput",
)
LOWER_IS_BETTER_PREFIXES = ("ttft", "tpot", "itl", "latency", "e2e_latency")


def _is_regression(metric_name: str, delta_percent: float) -> bool:
    """Is this delta a regression (bad) or improvement (good)?"""
    if delta_percent == 0:
        return False
    # Normalize name: strip common prefixes/suffixes
    name = metric_name.lower()
    if any(p in name for p in HIGHER_IS_BETTER_PREFIXES):
        return delta_percent < 0  # throughput dropped → bad
    if any(p in name for p in LOWER_IS_BETTER_PREFIXES):
        return delta_percent > 0  # latency rose → bad
    # Unknown metric direction — treat any |z| > threshold as regression
    return True


def _severity(z: float, delta_percent: float) -> str:
    absz = abs(z)
    absd = abs(delta_percent)
    if absz >= 4.0 or absd >= 10.0:
        return "critical"
    if absz >= 3.0 or absd >= 5.0:
        return "major"
    return "minor"


@dataclass(frozen=True)
class Detection:
    run_id: int
    metric_name: str
    severity: str
    delta_percent: float
    z_score: float


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _mad(values: Iterable[float], median: float) -> float:
    """Median Absolute Deviation."""
    deviations = [abs(v - median) for v in values]
    return statistics.median(deviations) if deviations else 0.0


def _fetch_history(
    conn: sqlite3.Connection,
    config_name: str,
    concurrency: int,
    metric_name: str,
    exclude_run_id: int,
    window_days: int,
) -> list[float]:
    """Get historical values of this metric for the same (config, conc)
    in the last `window_days`, excluding the current run being evaluated."""
    rows = conn.execute(
        """
        SELECT m.value
        FROM runs r
        JOIN metrics m ON m.run_id = r.id
        WHERE r.config_name = ?
          AND r.concurrency = ?
          AND m.name = ?
          AND r.status = 'passed'
          AND r.id != ?
          AND r.started_at > datetime('now', ?)
        """,
        (config_name, concurrency, metric_name, exclude_run_id, f"-{window_days} day"),
    ).fetchall()
    return [r["value"] for r in rows]


def detect_for_run(
    conn: sqlite3.Connection,
    run_id: int,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[Detection]:
    """Flag anomalies for every metric on the given run. Returns the list of
    detections (and writes them to the `regressions` table).
    """
    run_row = conn.execute(
        "SELECT config_name, concurrency FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    if run_row is None:
        return []

    config_name = run_row["config_name"]
    concurrency = run_row["concurrency"]

    metric_rows = conn.execute(
        "SELECT name, value FROM metrics WHERE run_id = ?", (run_id,)
    ).fetchall()

    detections: list[Detection] = []
    for mr in metric_rows:
        metric_name = mr["name"]
        current = mr["value"]

        history = _fetch_history(
            conn, config_name, concurrency, metric_name, run_id, window_days
        )
        if len(history) < MIN_SAMPLES:
            continue

        median = statistics.median(history)
        mad = _mad(history, median) * MAD_TO_SIGMA  # scaled to ~stddev
        if mad == 0 or math.isnan(mad):
            continue  # no variation — skip

        z = (current - median) / mad
        delta_percent = (current - median) / median * 100 if median != 0 else 0

        if abs(z) < Z_THRESHOLD:
            continue

        if not _is_regression(metric_name, delta_percent):
            # Improvement, not regression — don't flag as a regression
            continue

        severity = _severity(z, delta_percent)
        detections.append(
            Detection(
                run_id=run_id,
                metric_name=metric_name,
                severity=severity,
                delta_percent=delta_percent,
                z_score=z,
            )
        )

    for d in detections:
        conn.execute(
            """
            INSERT OR IGNORE INTO regressions (
                run_id, metric_name, severity, delta_percent, z_score,
                baseline_window_days, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                d.run_id,
                d.metric_name,
                d.severity,
                d.delta_percent,
                d.z_score,
                window_days,
                _now_iso(),
            ),
        )

    if detections:
        logger.info(
            "flagged %d regression(s) for run_id=%d (%s/%d)",
            len(detections),
            run_id,
            config_name,
            concurrency,
        )

    return detections


def resolve_stale_regressions(conn: sqlite3.Connection) -> int:
    """Auto-resolve regressions where a later passing run at the same
    (config, concurrency, metric) is no longer anomalous. Returns count
    of regressions resolved.
    """
    # For each active regression, find the next passing run at same coordinates
    # that doesn't itself have a regression on the same metric. If that exists,
    # the regression is considered "recovered".
    rows = conn.execute("""
        SELECT reg.id, reg.run_id, reg.metric_name,
               r.config_name, r.concurrency, r.started_at
        FROM regressions reg
        JOIN runs r ON r.id = reg.run_id
        WHERE reg.resolved_at IS NULL
        """).fetchall()

    resolved = 0
    now = _now_iso()
    for row in rows:
        nxt = conn.execute(
            """
            SELECT r.id
            FROM runs r
            LEFT JOIN regressions reg
                ON reg.run_id = r.id AND reg.metric_name = ?
            WHERE r.config_name = ?
              AND r.concurrency = ?
              AND r.started_at > ?
              AND r.status = 'passed'
              AND reg.id IS NULL
            ORDER BY r.started_at
            LIMIT 1
            """,
            (
                row["metric_name"],
                row["config_name"],
                row["concurrency"],
                row["started_at"],
            ),
        ).fetchone()

        if nxt is not None:
            conn.execute(
                """
                UPDATE regressions
                SET resolved_at = ?, resolution_run_id = ?
                WHERE id = ?
                """,
                (now, nxt["id"], row["id"]),
            )
            resolved += 1

    if resolved:
        logger.info("auto-resolved %d regression(s)", resolved)
    return resolved
