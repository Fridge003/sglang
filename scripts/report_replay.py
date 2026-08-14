#!/usr/bin/env python3
"""Compute centralized statistics from a DSV4 replay artifact set."""

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HISTOGRAM_UPPER_BOUNDS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
]


def percentile(values: Iterable[float], percent: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _number(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def load_requests(path: Path) -> List[Dict[str, Any]]:
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(f"line {line_number} has no request_id")
            if request_id in seen:
                raise ValueError(f"duplicate request_id: {request_id}")
            seen.add(request_id)
            records.append(record)
    if not records:
        raise ValueError("requests file is empty")
    return records


def _parse_iso8601(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("completed_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid completed_at: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("completed_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def infer_interval(record: Mapping[str, Any]) -> Tuple[datetime, datetime]:
    completed = _parse_iso8601(record.get("completed_at"))
    latency_s = _number(record, "latency_s")
    if latency_s <= 0:
        raise ValueError("latency_s must be greater than zero")
    return datetime.fromtimestamp(completed.timestamp() - latency_s, timezone.utc), completed


def request_metric_summary(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    successful = [record for record in records if record.get("success") is True]
    if not successful:
        raise ValueError("no successful requests")

    ttfts = []
    tpots = []
    decode_tps = []
    cache_hit_rates = []
    for record in successful:
        ttft_s = _number(record, "ttft_s")
        if ttft_s < 0:
            raise ValueError("ttft_s must not be negative")
        ttfts.append(ttft_s)

        if record.get("tpot_s") is not None:
            tpot_s = _number(record, "tpot_s")
            if tpot_s < 0:
                raise ValueError("tpot_s must not be negative")
            tpots.append(tpot_s)

        completion_tokens = int(_number(record, "completion_tokens"))
        latency_s = _number(record, "latency_s")
        decode_duration_s = latency_s - ttft_s
        if completion_tokens > 1 and decode_duration_s > 0:
            decode_tps.append((completion_tokens - 1) / decode_duration_s)

        if record.get("cached_tokens") is None:
            raise ValueError("cached_tokens is missing from a successful request")
        cached_tokens = _number(record, "cached_tokens")
        prompt_tokens = _number(record, "prompt_tokens")
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be greater than zero")
        cache_hit_rates.append(cached_tokens / prompt_tokens)

    def milliseconds(values: Sequence[float], percent: float) -> Optional[float]:
        value = percentile(values, percent)
        return None if value is None else value * 1000.0

    return {
        "p50_ttft_ms": milliseconds(ttfts, 50),
        "p99_ttft_ms": milliseconds(ttfts, 99),
        "p50_tpot_ms": milliseconds(tpots, 50),
        "p99_tpot_ms": milliseconds(tpots, 99),
        "p50_decode_tps_per_user": percentile(decode_tps, 50),
        "p50_cache_hit_rate": percentile(cache_hit_rates, 50),
        "p99_cache_hit_rate": percentile(cache_hit_rates, 99),
    }


def weighted_percentile(
    weight_by_value: Mapping[float, float], percent: float
) -> Optional[float]:
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    positive = sorted(
        (float(value), float(weight))
        for value, weight in weight_by_value.items()
        if float(weight) > 0
    )
    if not positive:
        return None
    total_weight = sum(weight for _, weight in positive)
    target = total_weight * percent / 100.0
    cumulative = 0.0
    for value, weight in positive:
        cumulative += weight
        if cumulative >= target:
            return value
    return positive[-1][0]


def _successful_intervals(
    records: Iterable[Mapping[str, Any]],
) -> List[Tuple[datetime, datetime]]:
    intervals = [
        infer_interval(record)
        for record in records
        if record.get("success") is True
    ]
    if not intervals:
        raise ValueError("no successful request intervals")
    return intervals


def concurrency_summary(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    intervals = _successful_intervals(records)
    deltas: Dict[datetime, int] = defaultdict(int)
    for started, completed in intervals:
        deltas[started] += 1
        deltas[completed] -= 1

    duration_by_concurrency: Dict[int, float] = defaultdict(float)
    active = 0
    previous = min(deltas)
    for timestamp in sorted(deltas):
        duration_s = (timestamp - previous).total_seconds()
        if duration_s > 0:
            duration_by_concurrency[active] += duration_s
        active += deltas[timestamp]
        if active < 0:
            raise ValueError("request interval sweep produced negative concurrency")
        previous = timestamp
    if active != 0:
        raise ValueError("request interval sweep did not return to zero")

    total_duration_s = sum(duration_by_concurrency.values())
    if total_duration_s <= 0:
        raise ValueError("replay interval must have positive duration")
    mean_concurrency = sum(
        concurrency * duration
        for concurrency, duration in duration_by_concurrency.items()
    ) / total_duration_s
    return {
        "p50_concurrency": weighted_percentile(duration_by_concurrency, 50),
        "p99_concurrency": weighted_percentile(duration_by_concurrency, 99),
        "mean_concurrency": mean_concurrency,
        "duration_s": total_duration_s,
        "seconds_by_concurrency": dict(sorted(duration_by_concurrency.items())),
    }


def qps_summary(
    records: Iterable[Mapping[str, Any]], window_s: float
) -> Dict[str, Any]:
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("QPS window must be a positive finite number")
    intervals = _successful_intervals(records)
    replay_start = min(started for started, _ in intervals)
    replay_end = max(completed for _, completed in intervals)
    duration_s = (replay_end - replay_start).total_seconds()
    window_count = max(1, math.ceil(duration_s / window_s))
    arrivals = [0] * window_count
    for started, _ in intervals:
        index = math.floor((started - replay_start).total_seconds() / window_s)
        arrivals[min(index, window_count - 1)] += 1
    qps_values = [count / window_s for count in arrivals]
    return {
        "p50_qps": percentile(qps_values, 50),
        "p99_qps": percentile(qps_values, 99),
        "window_s": window_s,
        "window_count": window_count,
    }


def build_histogram(
    values: Iterable[int], upper_bounds: Sequence[int]
) -> List[Dict[str, Any]]:
    bounds = [int(bound) for bound in upper_bounds]
    if not bounds or any(bound < 0 for bound in bounds):
        raise ValueError("histogram bounds must be non-negative")
    if bounds != sorted(set(bounds)):
        raise ValueError("histogram bounds must be strictly increasing")
    observed = []
    for value in values:
        numeric = int(value)
        if numeric < 0:
            raise ValueError("histogram values must be non-negative")
        observed.append(numeric)
    if not observed:
        raise ValueError("histogram values are empty")

    counts = [0] * (len(bounds) + 1)
    for value in observed:
        placed = False
        for index, bound in enumerate(bounds):
            if value <= bound:
                counts[index] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    total = len(observed)
    buckets = []
    cumulative = 0
    lower = 0
    for index, count in enumerate(counts):
        cumulative += count
        if index < len(bounds):
            upper = bounds[index]
            label = f"{lower}-{upper}"
            lower_bound: Optional[int] = lower
            upper_bound: Optional[int] = upper
            lower = upper + 1
        else:
            label = f">{bounds[-1]}"
            lower_bound = bounds[-1] + 1
            upper_bound = None
        buckets.append(
            {
                "range": label,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "count": count,
                "percentage": count / total * 100.0,
                "cumulative_percentage": cumulative / total * 100.0,
            }
        )
    return buckets


_ACCEPTANCE_PATTERN = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: [^\]]*)?\]"
    r".*?accept len:\s*(?P<value>\d+(?:\.\d+)?)"
)


def parse_acceptance_samples(
    path: Path, replay_start: datetime, replay_end: datetime
) -> List[float]:
    if replay_start.tzinfo is None or replay_end.tzinfo is None:
        raise ValueError("replay interval must be timezone-aware")
    start_utc = replay_start.astimezone(timezone.utc)
    end_utc = replay_end.astimezone(timezone.utc)
    if end_utc < start_utc:
        raise ValueError("replay end precedes replay start")

    samples = []
    with Path(path).open(encoding="utf-8", errors="replace") as source:
        for line in source:
            match = _ACCEPTANCE_PATTERN.search(line)
            if not match:
                continue
            timestamp = datetime.strptime(
                match.group("timestamp"), "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            if start_utc <= timestamp <= end_utc:
                samples.append(float(match.group("value")))
    return samples


def _replay_bounds(
    records: Iterable[Mapping[str, Any]],
) -> Tuple[datetime, datetime]:
    intervals = _successful_intervals(records)
    return (
        min(started for started, _ in intervals),
        max(completed for _, completed in intervals),
    )


def _summary_number(summary: Mapping[str, Any], key: str) -> Optional[float]:
    value = summary.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"summary {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"summary {key} must be finite")
    return result


def build_report(
    requests: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
    acceptance_samples: Sequence[float],
    qps_window_s: float,
    sources: Mapping[str, str],
) -> Dict[str, Any]:
    records = list(requests)
    successful = [record for record in records if record.get("success") is True]
    failed = len(records) - len(successful)
    expected_counts = {
        "total_requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": failed,
    }
    for key, expected in expected_counts.items():
        if key in summary and int(_summary_number(summary, key)) != expected:
            raise ValueError(
                f"summary {key}={summary[key]} does not match requests={expected}"
            )

    request_metrics = request_metric_summary(records)
    concurrency = concurrency_summary(records)
    qps = qps_summary(records, qps_window_s)
    replay_start, replay_end = _replay_bounds(records)
    replay_duration_s = _summary_number(summary, "replay_duration_s")
    if replay_duration_s is None:
        replay_duration_s = (replay_end - replay_start).total_seconds()

    warnings = []
    sample_values = [float(value) for value in acceptance_samples]
    if any(not math.isfinite(value) for value in sample_values):
        raise ValueError("acceptance samples must be finite")
    log_sample_mean = fmean(sample_values) if sample_values else None
    cumulative_mean = _summary_number(summary, "mean_accept_length")
    if cumulative_mean is not None:
        acceptance_mean = cumulative_mean
        mean_source = "summary"
    elif log_sample_mean is not None:
        acceptance_mean = log_sample_mean
        mean_source = "log_samples"
        warnings.append(
            "summary mean_accept_length was absent; mean uses server-log samples"
        )
    else:
        acceptance_mean = None
        mean_source = None
        warnings.append("no acceptance-length samples or cumulative mean were available")

    prompt_tokens = [
        int(_number(record, "prompt_tokens")) for record in successful
    ]
    completion_tokens = [
        int(_number(record, "completion_tokens")) for record in successful
    ]
    metrics = dict(request_metrics)
    metrics.update(
        {
            "p50_concurrency": concurrency["p50_concurrency"],
            "p99_concurrency": concurrency["p99_concurrency"],
            "mean_concurrency": concurrency["mean_concurrency"],
            "p50_qps": qps["p50_qps"],
            "p99_qps": qps["p99_qps"],
            "qps_window_s": qps["window_s"],
            "aggregate_decode_tps": _summary_number(summary, "decode_tps"),
        }
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "sources": dict(sources),
        "replay": {
            "start": replay_start.isoformat().replace("+00:00", "Z"),
            "end": replay_end.isoformat().replace("+00:00", "Z"),
            "duration_s": replay_duration_s,
        },
        "counts": {
            **expected_counts,
            "total_prompt_tokens": sum(prompt_tokens),
            "total_completion_tokens": sum(completion_tokens),
        },
        "metrics": metrics,
        "acceptance": {
            "p50": percentile(sample_values, 50),
            "p99": percentile(sample_values, 99),
            "mean": acceptance_mean,
            "mean_source": mean_source,
            "log_sample_mean": log_sample_mean,
            "sample_count": len(sample_values),
        },
        "isl_histogram": build_histogram(
            prompt_tokens, HISTOGRAM_UPPER_BOUNDS
        ),
        "osl_histogram": build_histogram(
            completion_tokens, HISTOGRAM_UPPER_BOUNDS
        ),
        "warnings": warnings,
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):,.6f}"


def _render_histogram(title: str, buckets: Sequence[Mapping[str, Any]]) -> str:
    maximum = max(int(bucket["count"]) for bucket in buckets) or 1
    lines = [
        f"## {title}",
        "",
        "| Range | Count | Percentage | Cumulative | Histogram |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for bucket in buckets:
        count = int(bucket["count"])
        width = 0 if count == 0 else max(1, round(count / maximum * 30))
        lines.append(
            "| {range} | {count:,} | {percentage:.4f}% | "
            "{cumulative:.4f}% | `{bar}` |".format(
                range=bucket["range"],
                count=count,
                percentage=float(bucket["percentage"]),
                cumulative=float(bucket["cumulative_percentage"]),
                bar="#" * width,
            )
        )
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    metrics = report["metrics"]
    acceptance = report["acceptance"]
    metric_rows = [
        ("P50 TTFT", metrics["p50_ttft_ms"], "ms"),
        ("P99 TTFT", metrics["p99_ttft_ms"], "ms"),
        ("P50 TPOT", metrics["p50_tpot_ms"], "ms/token"),
        ("P99 TPOT", metrics["p99_tpot_ms"], "ms/token"),
        (
            "P50 Decode TPS per user",
            metrics["p50_decode_tps_per_user"],
            "token/s",
        ),
        ("P50 concurrency", metrics["p50_concurrency"], "requests"),
        ("P99 concurrency", metrics["p99_concurrency"], "requests"),
        ("P50 QPS", metrics["p50_qps"], "requests/s"),
        ("P99 QPS", metrics["p99_qps"], "requests/s"),
        ("P50 cache hit rate", metrics["p50_cache_hit_rate"] * 100, "%"),
        ("P99 cache hit rate", metrics["p99_cache_hit_rate"] * 100, "%"),
    ]
    lines = [
        "# DSV4 replay report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Run summary",
        "",
        f"- Requests: {counts['successful_requests']:,} successful / "
        f"{counts['failed_requests']:,} failed / {counts['total_requests']:,} total",
        f"- Prompt tokens: {counts['total_prompt_tokens']:,}",
        f"- Completion tokens: {counts['total_completion_tokens']:,}",
        f"- Replay duration: {_format_number(report['replay']['duration_s'])} seconds",
        f"- QPS window: {_format_number(metrics['qps_window_s'])} seconds",
        "",
        "## Central metrics",
        "",
        "| Metric | Value | Unit |",
        "| --- | ---: | --- |",
    ]
    lines.extend(
        f"| {label} | {_format_number(value)} | {unit} |"
        for label, value, unit in metric_rows
    )
    lines.extend(
        [
            "",
            "## Acceptance Length",
            "",
            "| Metric | Value | Source |",
            "| --- | ---: | --- |",
            f"| P50 | {_format_number(acceptance['p50'])} | in-window server-log samples |",
            f"| P99 | {_format_number(acceptance['p99'])} | in-window server-log samples |",
            f"| Mean | {_format_number(acceptance['mean'])} | {acceptance['mean_source'] or 'unavailable'} |",
            f"| Log sample mean | {_format_number(acceptance['log_sample_mean'])} | "
            f"{acceptance['sample_count']:,} samples |",
            "",
            _render_histogram("ISL Histogram", report["isl_histogram"]),
            "",
            _render_histogram("OSL Histogram", report["osl_histogram"]),
            "",
            "## Source artifacts",
            "",
        ]
    )
    for name, path in sorted(report["sources"].items()):
        lines.append(f"- {name}: `{path}`")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- Per-user decode TPS is `(completion_tokens - 1) / (latency - TTFT)`.",
            "- Concurrency percentiles are time-weighted over the full inferred replay interval.",
            "- QPS is based on request arrivals in fixed non-overlapping windows, including empty windows.",
            "- Cache hit rate is `cached_tokens / prompt_tokens` per successful request.",
            "- Ordinary percentiles use linear interpolation at `(N - 1) * p / 100`.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            delete=False,
        ) as destination:
            temp_path = Path(destination.name)
            destination.write(text)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _load_summary(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("summary must be a JSON object")
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qps-window-s", type=float, default=1.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        requests = load_requests(args.requests)
        summary = _load_summary(args.summary)
        replay_start, replay_end = _replay_bounds(requests)
        acceptance_samples = parse_acceptance_samples(
            args.server_log, replay_start, replay_end
        )
        report = build_report(
            requests,
            summary,
            acceptance_samples,
            args.qps_window_s,
            {
                "requests": str(args.requests),
                "summary": str(args.summary),
                "server_log": str(args.server_log),
            },
        )
        _atomic_write_text(
            args.output_dir / "report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write_text(args.output_dir / "report.md", render_markdown(report))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output_dir / 'report.md'}")
    print(f"wrote {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
