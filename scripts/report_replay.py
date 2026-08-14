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
