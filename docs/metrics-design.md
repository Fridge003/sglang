# Central Replay Statistics and Publication Branch Design

## Goal

Add one dependency-free statistics command that consumes the replay artifacts,
computes the requested latency, throughput, load, cache, acceptance, and length
distributions, and emits a human-readable Markdown report plus a machine-readable
JSON report. Publish the command, replay tooling, tests, instructions, and the
generated report on a source-free branch in `Fridge003/sglang`.

Token-level ITL is explicitly out of scope because the completed replay did not
persist individual token timestamps and the user chose not to report ITL.

## Inputs and command

The command uses only the Python standard library:

```bash
python3 report_replay.py \
  --requests remote/attempt2/formal/requests.jsonl \
  --summary remote/attempt2/formal/summary.json \
  --server-log remote/attempt2/server.log \
  --output-dir reports/attempt2 \
  --qps-window-s 1
```

`--requests` is required. `--summary` supplies the authoritative replay duration
and cumulative mean speculative acceptance length. `--server-log` supplies
periodic acceptance-length samples. The script validates JSON types, required
fields, positive durations, unique request IDs, and the consistency of summary
request counts with the JSONL.

## Metric definitions

All request-level distributions include successful requests only and use linear
interpolation at positions `(N - 1) * percentile / 100`.

- **P50/P99 TTFT:** percentiles of `ttft_s`, reported in milliseconds.
- **P50/P99 TPOT:** percentiles of `tpot_s`, reported in milliseconds per
  token. Records without a defined TPOT are excluded.
- **P50 decode TPS per user:** percentile of
  `(completion_tokens - 1) / (latency_s - ttft_s)` for requests with at least
  two completion tokens and a positive decode duration.
- **P50/P99 cache hit rate:** percentiles of
  `cached_tokens / prompt_tokens`. Missing cache details are rejected instead
  of silently treated as zero because the formal replay enabled cache reports.
- **P50/P99 concurrency:** exact time-weighted distribution of active requests.
  Each request starts at `completed_at - latency_s` and ends at `completed_at`.
  A sweep over all start/end events produces intervals and their active counts.
  The distribution covers the complete interval from the first inferred start
  through the final completion, including any zero-concurrency gaps.
- **P50/P99 QPS:** request arrivals per non-overlapping window of
  `--qps-window-s`, anchored at the first inferred request start and spanning
  through the final completion. Empty windows are included, and counts are
  divided by the configured window duration.
- **P50/P99 acceptance length:** linear percentiles of `accept len:` samples in
  the server log whose timestamps fall inside the formal replay interval.
- **Mean acceptance length:** the final cumulative
  `summary.json["mean_accept_length"]`. If it is absent, the script falls back
  to the arithmetic mean of the in-window server-log samples and emits a
  warning. The report also records the sample count and sample arithmetic mean
  so the two sources remain auditable.
- **ISL/OSL histograms:** actual `prompt_tokens` and `completion_tokens` using
  fixed power-of-two upper bounds: 128, 256, 512, 1,024, 2,048, 4,096, 8,192,
  16,384, 32,768, 65,536, and 131,072, plus an overflow bucket. Each Markdown
  histogram row contains range, count, percentage, cumulative percentage, and
  a proportional ASCII bar. JSON preserves edges and exact counts.

The report also carries successful/failed request counts, prompt/completion
token totals, replay duration, QPS window size, source paths, and warnings.

## Formal replay time and log parsing

The replay start is inferred from the earliest request start and the end from
the latest `completed_at`. Server log lines begin with a bracketed timestamp
such as `[2026-08-14 04:16:35 TP0]`; those timestamps are interpreted as UTC to
match the replay JSONL. Only lines inside the inferred replay interval are used,
which excludes startup, smoke-test, and post-run messages. Lines without both a
parseable timestamp and `accept len:` are ignored.

## Outputs

`--output-dir` receives:

- `report.md`: concise metric tables, ISL/OSL histograms, provenance, metric
  definitions, sample counts, and warnings.
- `report.json`: the same data in stable structured form for automation.

Writes use temporary sibling files followed by atomic replacement so an
interrupted run does not leave a partial report.

## Files and tests

- `report_replay.py`: parsing, reducers, histogram construction, renderers, and
  CLI.
- `test_report_replay.py`: synthetic tests for percentile math, per-user decode
  TPS, weighted concurrency, QPS windows including zeros, formal-window
  acceptance parsing, histogram boundaries, validation, and end-to-end output.
- Existing `replay_trace.py` and its test remain the source of replay data.

Implementation follows red-green-refactor. The generated report is then
validated against the existing 8,020-row successful artifact and compared with
the already independently verified summary metrics.

## Publication branch

Create orphan branch `codex/dsv4-trace-replay-report` targeting the
`fridge003` remote. It contains no SGLang source and no files inherited from the
default branch. Its layout is:

```text
README.md
environment.txt
scripts/
  launch_sglang.sh
  replay_trace.py
  report_replay.py
  run_full_replay.sh
  run_smoke_replay.sh
tests/
  test_replay_trace.py
  test_report_replay.py
docs/
  replay-design.md
  metrics-design.md
reports/
  attempt2-report.md
  attempt2-report.json
```

The original trace, request JSONL, server logs, model data, and Python cache
files are excluded to keep the branch small and avoid publishing workload
content. README instructions state where users must supply the trace and result
artifacts. Before pushing, tests, Python compilation, shell syntax checks,
report regeneration, a tracked-file allowlist, and a sensitive/large-file scan
must all pass.
