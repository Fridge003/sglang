# DSV4 Replay Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a standard-library command that computes the approved DSV4 replay statistics and renders auditable Markdown/JSON reports.

**Architecture:** `scripts/report_replay.py` owns pure reducers for request metrics, time-weighted concurrency, fixed-window arrival QPS, acceptance-log filtering, and power-of-two histograms, plus a thin CLI and two atomic renderers. Synthetic unit tests exercise each reducer with hand-derived values; an end-to-end test invokes the real CLI. The existing 8,020-row artifact is then used to generate the checked-in report.

**Tech Stack:** Python 3 standard library, `unittest`, Bash, Git.

**Spec:** `docs/metrics-design.md`

## Global Constraints

- Do not report ITL.
- Use only the Python standard library.
- Use successful JSONL rows for request-level distributions and reject duplicate request IDs.
- Use linear interpolation at `(N - 1) * percentile / 100` for ordinary percentiles.
- Use exact time-weighted concurrency over inferred request intervals.
- Use 1-second non-overlapping request-arrival windows by default and include zero windows.
- Use formal-window server-log samples for acceptance P50/P99 and the summary cumulative value for mean acceptance length.
- Do not publish the original trace, request JSONL, server logs, model data, Python caches, or SGLang source.
- The publication branch is the orphan branch `codex/dsv4-trace-replay-report` targeting remote `fridge003`.

---

### Task 1: Pure statistics reducers

**Files:**
- Create: `scripts/report_replay.py`
- Create: `tests/test_report_replay.py`

**Interfaces:**
- Produces: `percentile(values, percent) -> float | None`
- Produces: `weighted_percentile(weight_by_value, percent) -> float | None`
- Produces: `load_requests(path) -> list[dict]`
- Produces: `infer_interval(record) -> tuple[datetime, datetime]`
- Produces: `request_metric_summary(records) -> dict`
- Produces: `concurrency_summary(records) -> dict`
- Produces: `qps_summary(records, window_s) -> dict`
- Produces: `build_histogram(values, upper_bounds) -> list[dict]`
- Produces: `parse_acceptance_samples(path, start, end) -> list[float]`

- [ ] **Step 1: Add failing percentile and request-metric tests**

Create tests with literal expectations:

```python
def test_request_metrics_use_decode_intervals_and_actual_cache_usage(self):
    rows = [
        make_row("a", 0, 4, ttft_s=1, prompt=100, output=7, cached=80),
        make_row("b", 1, 3, ttft_s=0.5, prompt=200, output=6, cached=100),
    ]
    got = report_replay.request_metric_summary(rows)
    self.assertEqual(got["p50_ttft_ms"], 750.0)
    self.assertAlmostEqual(got["p50_decode_tps_per_user"], 8 / 3)
    self.assertEqual(got["p50_cache_hit_rate"], 0.65)
```

The first user decode rate is `(7 - 1) / (4 - 1) = 2`; the second is
`(6 - 1) / (2 - 0.5) = 10/3`; their linearly interpolated median is `8/3`,
which makes a wrong full-latency formula fail.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 tests/test_report_replay.py RequestMetricTest -v
```

Expected: import failure because `scripts/report_replay.py` does not exist.

- [ ] **Step 3: Implement percentile, loading, intervals, and request metrics**

Implement strict finite-number checks, unique-ID validation, successful-row
selection, milliseconds conversion, and these exact output keys:

```python
{
    "p50_ttft_ms": ...,
    "p99_ttft_ms": ...,
    "p50_tpot_ms": ...,
    "p99_tpot_ms": ...,
    "p50_decode_tps_per_user": ...,
    "p50_cache_hit_rate": ...,
    "p99_cache_hit_rate": ...,
}
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2. Expected: all request-metric tests pass.

- [ ] **Step 5: Add failing concurrency and QPS tests**

Use requests active over `[0,1]` and `[3,4]` to prove the concurrency sweep
includes the two-second zero gap: P50 concurrency is `0`, P99 is `1`. Use
arrival counts `[2, 0, 1, 0]` over four 1-second windows to prove P50 QPS is
`0.5` and P99 QPS is `1.97`.

- [ ] **Step 6: Run the load tests and verify RED**

```bash
python3 tests/test_report_replay.py LoadMetricTest -v
```

Expected: missing reducer functions because that load-metric behavior is not implemented.

- [ ] **Step 7: Implement exact sweep-line concurrency and fixed-window QPS**

Represent concurrency as `dict[int, float]` mapping active count to seconds.
Sort start/end events, accumulate the prior active count over each non-zero
interval, and then apply a cumulative-duration weighted percentile. Anchor QPS
bins at the earliest inferred start, include bins through the latest completion,
and divide counts by `window_s`.

- [ ] **Step 8: Add histogram and acceptance parsing tests, then verify RED**

Test values `[0, 1, 128, 129, 256, 257]` against bounds `[128, 256]` and
expect bucket counts `[3, 2, 1]`. Write a temporary log with acceptance samples
before, inside, and after a four-second formal interval; expect only the two
inside samples.

- [ ] **Step 9: Implement histogram and acceptance parsing, then verify GREEN**

Histogram buckets are inclusive at each upper bound and the overflow bucket is
strictly greater than the last bound. Parse log timestamps as UTC and match the
numeric value after `accept len:`. Run:

```bash
python3 tests/test_report_replay.py -v
```

Expected: all reducer tests pass.

- [ ] **Step 10: Commit the reducer slice**

```bash
git add scripts/report_replay.py tests/test_report_replay.py
git commit -m "feat: add replay statistics reducers"
```

### Task 2: CLI, renderers, and real report

**Files:**
- Modify: `scripts/report_replay.py`
- Modify: `tests/test_report_replay.py`
- Create: `reports/attempt2-report.md`
- Create: `reports/attempt2-report.json`

**Interfaces:**
- Consumes: all Task 1 reducers
- Produces: `build_report(requests, summary, acceptance_samples, qps_window_s, sources) -> dict`
- Produces: `render_markdown(report) -> str`
- Produces: CLI flags `--requests`, `--summary`, `--server-log`, `--output-dir`, and `--qps-window-s`

- [ ] **Step 1: Add a failing end-to-end CLI test**

Create temporary JSONL, summary, and server-log fixtures, invoke the actual
script with `subprocess.run`, and assert exit code zero plus both output files.
Parse `report.json` and assert the literal metric values; assert `report.md`
contains metric, ISL histogram, OSL histogram, provenance, and no ITL section.

- [ ] **Step 2: Run the CLI test and verify RED**

```bash
python3 tests/test_report_replay.py CliTest -v
```

Expected: failure because the CLI and renderers are not implemented.

- [ ] **Step 3: Implement report assembly and atomic JSON/Markdown writers**

The report schema is:

```python
{
    "schema_version": 1,
    "generated_at": "...Z",
    "sources": {...},
    "counts": {...},
    "metrics": {...},
    "acceptance": {
        "p50": ...,
        "p99": ...,
        "mean": ...,
        "mean_source": "summary" | "log_samples",
        "log_sample_mean": ...,
        "sample_count": ...,
    },
    "isl_histogram": [...],
    "osl_histogram": [...],
    "warnings": [...],
}
```

Write `report.json` and `report.md` to temporary sibling paths, flush and close,
then replace the destination paths atomically.

- [ ] **Step 4: Run all report tests and verify GREEN**

```bash
python3 tests/test_report_replay.py -v
python3 -m py_compile scripts/report_replay.py tests/test_report_replay.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 5: Generate the real report**

```bash
python3 scripts/report_replay.py \
  --requests "/Users/baizhou.zhang/Desktop/dsv4 optimize/replay_e3660f54/remote/attempt2/formal/requests.jsonl" \
  --summary "/Users/baizhou.zhang/Desktop/dsv4 optimize/replay_e3660f54/remote/attempt2/formal/summary.json" \
  --server-log "/Users/baizhou.zhang/Desktop/dsv4 optimize/replay_e3660f54/remote/attempt2/server.log" \
  --output-dir reports \
  --qps-window-s 1
mv reports/report.md reports/attempt2-report.md
mv reports/report.json reports/attempt2-report.json
```

- [ ] **Step 6: Validate the real report against independent facts**

Assert 8,020 requests, zero failures, 180,742,867 prompt tokens, 12,413,261
completion tokens, P50 TTFT `536.0490345337894` ms, P99 TTFT
`2106.461418156986` ms, P50 TPOT `40.97875800398908` ms, P99 TPOT
`94.65123315860394` ms, P50 cache rate `0.9580432310885934`, and P99 cache
rate `0.9964826631153246`. Assert acceptance sample count is positive and all
histogram bucket counts sum to 8,020.

- [ ] **Step 7: Commit the CLI and report slice**

```bash
git add scripts/report_replay.py tests/test_report_replay.py reports/
git commit -m "feat: generate centralized replay report"
```

### Task 3: Reproduction package and publication

**Files:**
- Create: `README.md`
- Create: `environment.txt`
- Create: `scripts/launch_sglang.sh`
- Create: `scripts/replay_trace.py`
- Create: `scripts/run_full_replay.sh`
- Create: `scripts/run_smoke_replay.sh`
- Create: `tests/test_replay_trace.py`
- Create: `docs/replay-design.md`

**Interfaces:**
- Consumes: the locally verified replay assets and Task 2 report command
- Produces: a source-free, runnable branch with exact launch/replay/report commands

- [ ] **Step 1: Copy only the approved reproduction assets**

Mechanically copy the verified scripts and documents from
`/Users/baizhou.zhang/Desktop/dsv4 optimize/replay_e3660f54`, preserving shell
executable bits. Do not copy `session__e3660f54.zip`, `remote/`, `dry-run/`,
`__pycache__/`, checksums for excluded files, or the nested publication clone.

- [ ] **Step 2: Write the branch README**

Document prerequisites, exact SGLang commit/environment, server launch, artifact
placement, smoke replay, full replay, report generation, metric formulas, output
files, the omission of ITL, and the fact that raw workload artifacts are not in
Git.

- [ ] **Step 3: Run the complete package verification**

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/replay_trace.py scripts/report_replay.py tests/test_replay_trace.py tests/test_report_replay.py
bash -n scripts/launch_sglang.sh scripts/run_smoke_replay.sh scripts/run_full_replay.sh
```

Expected: all tests pass; Python and Bash syntax checks exit zero.

- [ ] **Step 4: Enforce the tracked-file allowlist and size/sensitive scan**

The tracked paths must be only `README.md`, `environment.txt`, `scripts/`,
`tests/`, `docs/`, and `reports/`. Fail if any tracked file is larger than 1
MiB or if a tracked path matches `session__`, `requests.jsonl`, `server.log`,
`__pycache__`, `python/sglang`, or `sglang/`.

- [ ] **Step 5: Commit the reproduction package**

```bash
git add README.md environment.txt scripts tests docs reports
git commit -m "docs: package DSV4 trace replay reproduction"
```

- [ ] **Step 6: Run final verification and push**

Repeat Steps 3 and 4 against the committed tree, verify `git status --short` is
empty, then run:

```bash
git push -u origin codex/dsv4-trace-replay-report
git ls-remote --exit-code --heads origin refs/heads/codex/dsv4-trace-replay-report
```

Expected: push succeeds and the exact remote branch ref exists.
