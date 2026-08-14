#!/usr/bin/env python3
"""Replay Aster/TerminalBench trajectories against an OpenAI chat endpoint."""

import argparse
import asyncio
import json
import math
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


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


def cache_hit_rate(
    cached_tokens: Optional[int], prompt_tokens: Optional[int]
) -> Optional[float]:
    if cached_tokens is None or not prompt_tokens:
        return None
    return float(cached_tokens) / float(prompt_tokens)


def normalize_cached_tokens(
    cached_tokens: Optional[int],
    prompt_tokens: Optional[int],
    cache_report_enabled: bool,
) -> Optional[int]:
    # SGLang omits prompt_tokens_details when the reported cached count is zero.
    if cached_tokens is None and prompt_tokens is not None and cache_report_enabled:
        return 0
    return cached_tokens


def _text(content: Optional[Dict[str, Any]]) -> str:
    if not content:
        return ""
    return "".join(
        str(part.get("text", ""))
        for part in content.get("parts", [])
        if isinstance(part, dict)
    )


def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []
    for tool in tools:
        function = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object"}),
        }
        if "strict" in tool:
            function["strict"] = tool["strict"]
        converted.append({"type": "function", "function": function})
    return converted


def _assistant_message(event: Dict[str, Any]) -> Dict[str, Any]:
    message = {
        "role": "assistant",
        "content": _text(event.get("content")),
    }
    reasoning = event.get("reasoning")
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    calls = []
    for call in event.get("tool_calls") or []:
        arguments = call.get("raw_arguments")
        if arguments is None:
            arguments = json.dumps(
                call.get("arguments", {}), ensure_ascii=False, separators=(",", ":")
            )
        calls.append(
            {
                "id": call["call_id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": arguments},
            }
        )
    if calls:
        message["tool_calls"] = calls
    return message


def iter_replay_requests(
    sequence: Dict[str, Any], trajectory_id: str, start_offset_s: float
) -> Iterator[Dict[str, Any]]:
    events = sequence.get("events", [])
    tools = _convert_tools(sequence.get("tools", []))
    history = []
    pending_tool_duration_s = 0.0

    for event_index, event in enumerate(events):
        kind = event.get("kind")
        if kind == "system_message":
            history.append({"role": "system", "content": _text(event.get("content"))})
        elif kind == "user_message":
            history.append({"role": "user", "content": _text(event.get("content"))})
        elif kind == "agent_message":
            usage = (event.get("extra") or {}).get("usage") or {}
            yield {
                "request_id": "%s:%d" % (trajectory_id, event_index),
                "trajectory_id": trajectory_id,
                "event_index": event_index,
                "start_offset_s": float(start_offset_s),
                "think_time_before_s": pending_tool_duration_s,
                "messages": list(history),
                "tools": tools,
                "max_tokens": int(usage.get("completion_tokens") or 1),
                "trace_prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "trace_completion_tokens": int(usage.get("completion_tokens") or 0),
                "trace_cached_tokens": int(usage.get("cached_tokens") or 0),
                "trace_finish_reason": event.get("finish_reason"),
            }
            pending_tool_duration_s = 0.0
            history.append(_assistant_message(event))
        elif kind == "tool_result":
            result = event.get("result") or {}
            duration_ms = result.get("duration_ms")
            if isinstance(duration_ms, (int, float)):
                pending_tool_duration_s += float(duration_ms) / 1000.0
            call_ref = event.get("call_ref") or {}
            ref_index = call_ref.get("index")
            call_index = int(event.get("call_index") or 0)
            call_id = "unknown-tool-call"
            if isinstance(ref_index, int) and 0 <= ref_index < len(events):
                calls = events[ref_index].get("tool_calls") or []
                if 0 <= call_index < len(calls):
                    call_id = calls[call_index].get("call_id", call_id)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _text(result.get("content")),
                }
            )


def build_replay_requests(
    sequence: Dict[str, Any], trajectory_id: str, start_offset_s: float
) -> List[Dict[str, Any]]:
    return list(iter_replay_requests(sequence, trajectory_id, start_offset_s))


def build_payload(
    request: Dict[str, Any], model: str, respect_eos: bool = False
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": request["messages"],
        "max_tokens": request["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_cached_tokens_details": True,
        "ignore_eos": not respect_eos,
    }
    if request.get("tools"):
        payload["tools"] = request["tools"]
        payload["tool_choice"] = "auto"
    return payload


def feed_sse_bytes(state: bytearray, chunk: bytes) -> List[Dict[str, Any]]:
    state.extend(chunk.replace(b"\r\n", b"\n"))
    payloads = []
    while True:
        boundary = state.find(b"\n\n")
        if boundary < 0:
            break
        frame = bytes(state[:boundary])
        del state[: boundary + 2]
        data_lines = []
        for line in frame.splitlines():
            if line.startswith(b"data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        data = b"\n".join(data_lines)
        if data == b"[DONE]":
            continue
        payloads.append(json.loads(data.decode("utf-8")))
    return payloads


class StreamAccumulator:
    def __init__(self) -> None:
        self.ttft_s = None
        self.prompt_tokens = None
        self.completion_tokens = None
        self.cached_tokens = None
        self.finish_reason = None

    def update(self, payload: Dict[str, Any], elapsed_s: float) -> None:
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            has_token = bool(
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("tool_calls")
            )
            if has_token and self.ttft_s is None:
                self.ttft_s = float(elapsed_s)
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]

        usage = payload.get("usage")
        if usage:
            self.prompt_tokens = usage.get("prompt_tokens")
            self.completion_tokens = usage.get("completion_tokens")
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens")
            if cached is None:
                cached = usage.get("cached_tokens")
            if cached is None:
                extension = usage.get("sglang") or payload.get("sglang") or {}
                cached_details = extension.get("cached_tokens_details") or {}
                cached = cached_details.get("device")
            self.cached_tokens = cached

    def finish(self, total_latency_s: float) -> Dict[str, Any]:
        completion_tokens = int(self.completion_tokens or 0)
        tpot_s = None
        if self.ttft_s is not None and completion_tokens > 1:
            tpot_s = (float(total_latency_s) - self.ttft_s) / (
                completion_tokens - 1
            )
        return {
            "ttft_s": self.ttft_s,
            "tpot_s": tpot_s,
            "latency_s": float(total_latency_s),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "finish_reason": self.finish_reason,
        }


def summarize_records(
    records: Iterable[Dict[str, Any]],
    replay_duration_s: float,
    mean_accept_length: Optional[float],
) -> Dict[str, Any]:
    all_records = list(records)
    successful = [record for record in all_records if record.get("success")]
    ttfts = [record["ttft_s"] for record in successful if record.get("ttft_s") is not None]
    tpots = [record["tpot_s"] for record in successful if record.get("tpot_s") is not None]
    hit_rates = [
        rate
        for rate in (
            cache_hit_rate(record.get("cached_tokens"), record.get("prompt_tokens"))
            for record in successful
        )
        if rate is not None
    ]
    prompt_with_cache = [
        record
        for record in successful
        if record.get("cached_tokens") is not None and record.get("prompt_tokens")
    ]
    total_prompt_for_cache = sum(
        int(record["prompt_tokens"]) for record in prompt_with_cache
    )
    total_cached = sum(int(record["cached_tokens"]) for record in prompt_with_cache)
    total_completion = sum(int(record.get("completion_tokens") or 0) for record in successful)

    def milliseconds(value: Optional[float]) -> Optional[float]:
        return None if value is None else value * 1000.0

    return {
        "total_requests": len(all_records),
        "successful_requests": len(successful),
        "failed_requests": len(all_records) - len(successful),
        "replay_duration_s": float(replay_duration_s),
        "total_prompt_tokens": sum(
            int(record.get("prompt_tokens") or 0) for record in successful
        ),
        "total_completion_tokens": total_completion,
        "p50_ttft_ms": milliseconds(percentile(ttfts, 50)),
        "p99_ttft_ms": milliseconds(percentile(ttfts, 99)),
        "p50_tpot_ms": milliseconds(percentile(tpots, 50)),
        "p99_tpot_ms": milliseconds(percentile(tpots, 99)),
        "decode_tps": (
            total_completion / replay_duration_s if replay_duration_s > 0 else None
        ),
        "p50_cache_hit_rate": percentile(hit_rates, 50),
        "p99_cache_hit_rate": percentile(hit_rates, 99),
        "token_weighted_cache_hit_rate": (
            total_cached / total_prompt_for_cache if total_prompt_for_cache else None
        ),
        "requests_with_cache_details": len(hit_rates),
        "mean_accept_length": mean_accept_length,
    }


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def discover_trajectories(trace_path: Path) -> List[Dict[str, Any]]:
    discovered = []
    with zipfile.ZipFile(str(trace_path)) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.endswith("/agent/trajectory.json")
        )
        for member in members:
            trajectory = json.loads(archive.read(member))
            sequence = next(iter(trajectory["sequences"].values()))
            lifecycle = sequence.get("lifecycle") or trajectory.get("lifecycle") or {}
            started_at = _parse_timestamp(lifecycle["started_at"])
            request_count = sum(
                event.get("kind") == "agent_message"
                for event in sequence.get("events", [])
            )
            discovered.append(
                {
                    "member": member,
                    "trajectory_id": trajectory.get("trajectory_id")
                    or sequence.get("sequence_id")
                    or member,
                    "started_at": started_at,
                    "request_count": request_count,
                }
            )
    if not discovered:
        raise ValueError("no agent/trajectory.json files found in trace archive")
    earliest = min(item["started_at"] for item in discovered)
    for item in discovered:
        item["start_offset_s"] = (item["started_at"] - earliest).total_seconds()
    return sorted(discovered, key=lambda item: (item["start_offset_s"], item["member"]))


def load_sequence(trace_path: Path, member: str) -> Dict[str, Any]:
    with zipfile.ZipFile(str(trace_path)) as archive:
        trajectory = json.loads(archive.read(member))
    return next(iter(trajectory["sequences"].values()))


def _root_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized[:-3] if normalized.endswith("/v1") else normalized


def _extract_mean_accept_length(server_info: Dict[str, Any]) -> Optional[float]:
    candidates = [server_info]
    decode = server_info.get("decode")
    if isinstance(decode, list):
        candidates.extend(item for item in decode if isinstance(item, dict))
    for candidate in candidates:
        states = candidate.get("internal_states")
        if isinstance(states, list):
            for state in states:
                if isinstance(state, dict) and "avg_spec_accept_length" in state:
                    return float(state["avg_spec_accept_length"])
    return None


async def _read_json_response(response: Any) -> Any:
    text = await response.text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


async def perform_request(
    session: Any,
    endpoint: str,
    payload: Dict[str, Any],
    timeout_s: float,
    cache_report_enabled: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    accumulator = StreamAccumulator()
    sse_state = bytearray()
    timeout = session.timeout.__class__(total=timeout_s)
    try:
        async with session.post(endpoint, json=payload, timeout=timeout) as response:
            if response.status != 200:
                body = await response.text()
                return {
                    "success": False,
                    "status_code": response.status,
                    "error": body[:16000],
                    "latency_s": time.perf_counter() - started,
                }
            async for chunk in response.content.iter_chunked(65536):
                for data in feed_sse_bytes(sse_state, chunk):
                    accumulator.update(data, time.perf_counter() - started)
        result = accumulator.finish(time.perf_counter() - started)
        result["cached_tokens"] = normalize_cached_tokens(
            result.get("cached_tokens"),
            result.get("prompt_tokens"),
            cache_report_enabled,
        )
        result.update({"success": True, "status_code": 200})
        return result
    except Exception as exc:
        return {
            "success": False,
            "status_code": None,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "latency_s": time.perf_counter() - started,
        }


async def _post_maintenance(session: Any, url: str, body: Optional[dict] = None) -> Any:
    async with session.post(url, json=body) as response:
        data = await _read_json_response(response)
        if response.status != 200:
            raise RuntimeError("POST %s failed (%d): %r" % (url, response.status, data))
        return data


async def _get_json(session: Any, url: str) -> Dict[str, Any]:
    async with session.get(url) as response:
        data = await _read_json_response(response)
        if response.status != 200:
            raise RuntimeError("GET %s failed (%d): %r" % (url, response.status, data))
        return data


def _load_existing_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSONL at line %d: %s" % (line_number, exc))
    return records


async def run_replay(args: argparse.Namespace) -> Dict[str, Any]:
    trace_path = Path(args.trace).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "requests.jsonl"
    summary_path = output_dir / "summary.json"
    trajectories = discover_trajectories(trace_path)
    if args.limit_trajectories:
        trajectories = trajectories[: args.limit_trajectories]

    planned_requests = sum(item["request_count"] for item in trajectories)
    print(
        "Discovered %d trajectories and %d requests; original start span %.3fs"
        % (len(trajectories), planned_requests, trajectories[-1]["start_offset_s"]),
        flush=True,
    )
    if args.dry_run:
        return {
            "trajectory_count": len(trajectories),
            "planned_requests": planned_requests,
            "original_start_span_s": trajectories[-1]["start_offset_s"],
        }

    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("aiohttp is required: python3 -m pip install aiohttp") from exc

    existing_records = _load_existing_records(records_path) if args.resume else []
    if records_path.exists() and not args.resume and not args.force:
        raise FileExistsError(
            "%s exists; pass --force to overwrite or --resume" % records_path
        )
    completed_ids = {
        record.get("request_id") for record in existing_records if record.get("request_id")
    }
    mode = "a" if args.resume else "w"
    output_handle = records_path.open(mode, encoding="utf-8")

    timeout = aiohttp.ClientTimeout(total=None, connect=300, sock_read=None)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    root_url = _root_url(args.base_url)
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.max_in_flight) if args.max_in_flight else None
    records = list(existing_records)
    progress = {"done": len(existing_records), "failed": sum(not r.get("success") for r in existing_records)}
    formal_started_wall = datetime.now(timezone.utc).isoformat()
    replay_started = time.perf_counter()
    request_budget = {"claimed": 0}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.flush_before_run:
            response = await _post_maintenance(session, root_url + "/flush_cache")
            print("flush_cache:", response, flush=True)
        if args.reset_spec_before_run:
            response = await _post_maintenance(
                session, root_url + "/set_internal_state", {"server_args": {}}
            )
            print("reset_spec_counters:", response, flush=True)
        initial_server_info = await _get_json(session, root_url + "/server_info")
        cache_report_enabled = bool(
            initial_server_info.get("enable_cache_report", False)
        )
        if not cache_report_enabled:
            raise RuntimeError(
                "server cache reporting is disabled; launch SGLang with "
                "--enable-cache-report"
            )

        async def record_result(record: Dict[str, Any]) -> None:
            async with lock:
                records.append(record)
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                progress["done"] += 1
                if not record.get("success"):
                    progress["failed"] += 1
                if progress["done"] % args.progress_every == 0 or not record.get("success"):
                    elapsed = time.perf_counter() - replay_started
                    print(
                        "progress=%d/%d failed=%d elapsed=%.1fs"
                        % (progress["done"], planned_requests, progress["failed"], elapsed),
                        flush=True,
                    )
            if args.fail_fast and not record.get("success"):
                raise RuntimeError(
                    "fail-fast: request %s failed: %s"
                    % (record.get("request_id"), record.get("error"))
                )

        async def trajectory_worker(entry: Dict[str, Any]) -> None:
            target = replay_started + entry["start_offset_s"] / args.time_scale
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            sequence = load_sequence(trace_path, entry["member"])
            for request in iter_replay_requests(
                sequence, entry["trajectory_id"], entry["start_offset_s"]
            ):
                if request["request_id"] in completed_ids:
                    continue
                if request["think_time_before_s"] > 0:
                    await asyncio.sleep(request["think_time_before_s"] / args.time_scale)
                async with lock:
                    if args.limit_requests and request_budget["claimed"] >= args.limit_requests:
                        return
                    request_budget["claimed"] += 1
                if args.max_output_tokens:
                    request["max_tokens"] = min(
                        request["max_tokens"], args.max_output_tokens
                    )
                payload = build_payload(request, args.model, args.respect_eos)
                queued_at = time.perf_counter()
                if semaphore is None:
                    result = await perform_request(
                        session,
                        endpoint,
                        payload,
                        args.request_timeout,
                        cache_report_enabled,
                    )
                    queue_delay_s = 0.0
                else:
                    async with semaphore:
                        queue_delay_s = time.perf_counter() - queued_at
                        result = await perform_request(
                            session,
                            endpoint,
                            payload,
                            args.request_timeout,
                            cache_report_enabled,
                        )
                result.update(
                    {
                        "request_id": request["request_id"],
                        "trajectory_id": request["trajectory_id"],
                        "event_index": request["event_index"],
                        "trajectory_start_offset_s": request["start_offset_s"],
                        "think_time_before_s": request["think_time_before_s"],
                        "queue_delay_s": queue_delay_s,
                        "trace_prompt_tokens": request["trace_prompt_tokens"],
                        "trace_completion_tokens": request["trace_completion_tokens"],
                        "trace_cached_tokens": request["trace_cached_tokens"],
                        "requested_max_tokens": request["max_tokens"],
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                await record_result(result)

        tasks = [asyncio.create_task(trajectory_worker(entry)) for entry in trajectories]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            output_handle.flush()
            output_handle.close()

        replay_duration_s = time.perf_counter() - replay_started
        server_info = await _get_json(session, root_url + "/server_info")
        mean_accept_length = _extract_mean_accept_length(server_info)

    summary = summarize_records(records, replay_duration_s, mean_accept_length)
    summary.update(
        {
            "trace_path": str(trace_path),
            "model": args.model,
            "base_url": args.base_url,
            "trajectory_count": len(trajectories),
            "planned_requests": planned_requests,
            "original_start_span_s": trajectories[-1]["start_offset_s"],
            "time_scale": args.time_scale,
            "respect_eos": args.respect_eos,
            "max_output_tokens": args.max_output_tokens,
            "formal_started_at": formal_started_wall,
            "formal_completed_at": datetime.now(timezone.utc).isoformat(),
            "server_info": server_info,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="Path to session zip")
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument(
        "--model", default="deepseek-ai/DeepSeek-V4-Flash-0731"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--limit-trajectories", type=int)
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--respect-eos", action="store_true")
    parser.add_argument("--flush-before-run", action="store_true")
    parser.add_argument("--reset-spec-before-run", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Cancel outstanding trajectory workers after the first failed request",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.time_scale <= 0:
        raise SystemExit("--time-scale must be greater than zero")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be greater than zero")
    try:
        asyncio.run(run_replay(args))
    except KeyboardInterrupt:
        print("interrupted; completed JSONL records are durable", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
