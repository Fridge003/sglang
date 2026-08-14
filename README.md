# DeepSeek-V4 B200 trace replay

This orphan branch is a source-free reproduction package for replaying the
`session__e3660f54` workload against SGLang and producing one centralized
performance report. It intentionally contains no SGLang source, model data,
original trace, request JSONL, or server log.

The checked-in report was produced with:

- 8 x NVIDIA B200
- `deepseek-ai/DeepSeek-V4-Flash-0731`
- SGLang commit `8bbca87780d1a075dd54d1a5ca357760a4904bbe`
- `--mem-fraction-static 0.85`
- 267 trajectories and 8,020 requests at the trace's original time scale

## Checked-in result

| Metric | Result |
| --- | ---: |
| P50 TTFT | 536.049 ms |
| P99 TTFT | 2,106.461 ms |
| P50 TPOT | 40.979 ms/token |
| P99 TPOT | 94.651 ms/token |
| P50 decode TPS per user | 24.403 token/s |
| P50 concurrency | 21 requests |
| P99 concurrency | 65 requests |
| P50 QPS, 1-second arrival windows | 0 requests/s |
| P99 QPS, 1-second arrival windows | 3 requests/s |
| P50 cache hit rate | 95.8043% |
| P99 cache hit rate | 99.6483% |
| P50 acceptance length | 3.33 tokens/step |
| P99 acceptance length | 4.8852 tokens/step |
| Mean acceptance length | 3.64024 tokens/step |

See [the complete Markdown report](reports/attempt2-report.md) for the ISL and
OSL histograms. The same data is available in
[`reports/attempt2-report.json`](reports/attempt2-report.json).

## Package layout

```text
scripts/replay_trace.py       trace parser, scheduler, streaming client
scripts/report_replay.py      centralized statistics and report generator
scripts/launch_sglang.sh      8xB200 SGLang launch command
scripts/run_smoke_replay.sh   one-request compatibility run
scripts/run_full_replay.sh    cache-flushed formal replay
tests/                        replay and report behavior tests
docs/replay-design.md         replay fidelity and metric semantics
docs/metrics-design.md        centralized-report design
environment.txt               exact successful-run versions
reports/                      checked-in Markdown and JSON reports
```

## Prerequisites

- A Linux devbox with 8 NVIDIA B200 GPUs and enough local/shared storage for the
  model.
- Python 3.9 or newer.
- SGLang and dependencies listed in `environment.txt`.
- The original `session__e3660f54.zip`, supplied separately.

The recorded run used this SGLang revision:

```bash
cd /sgl-workspace/sglang
git fetch origin main
git checkout 8bbca87780d1a075dd54d1a5ca357760a4904bbe
SGLANG_BUILD_RUST_EXTS=none python3 -m pip install --upgrade \
  --extra-index-url https://download.pytorch.org/whl/cu130 -e python
python3 -m pip install --force-reinstall --no-deps \
  flashinfer-cubin==0.6.17 --index-url https://flashinfer.ai/whl
python3 -m pip install --force-reinstall --no-deps \
  'flashinfer-jit-cache==0.6.17+cu130' \
  --index-url https://flashinfer.ai/whl/cu130
```

## Launch SGLang

Copy the replay scripts and trace into a dedicated directory on the B200 host:

```bash
artifact_dir=/scratch/dsv4_trace_replay
mkdir -p "$artifact_dir"
cp scripts/replay_trace.py scripts/launch_sglang.sh \
  scripts/run_smoke_replay.sh scripts/run_full_replay.sh "$artifact_dir/"
cp /path/to/session__e3660f54.zip "$artifact_dir/"

"$artifact_dir/launch_sglang.sh" "$artifact_dir"
until curl -fsS http://localhost:30000/health; do sleep 10; done
```

The launch command is the requested DSPARK/FlashInfer MXFP4 configuration with
`--mem-fraction-static 0.85`. It also includes `--enable-cache-report`, which is
required to return scheduler-computed cached-token counts through the
OpenAI-compatible response. No allocator override is used.

## Smoke and formal replay

```bash
cd "$artifact_dir"
./run_smoke_replay.sh session__e3660f54.zip smoke
./run_full_replay.sh session__e3660f54.zip formal
```

The formal wrapper flushes the radix cache, resets speculative counters, uses
the original trajectory start offsets and tool-duration sleeps, requests each
recorded output length with EOS ignored, and stops at the first request failure.
At the original time scale the run can take more than five hours.

The formal output contains:

```text
formal/requests.jsonl
formal/summary.json
formal.console.log
server.log
```

## Generate the centralized report

Keep raw artifacts outside this Git checkout, then run:

```bash
python3 scripts/report_replay.py \
  --requests ../artifacts/formal/requests.jsonl \
  --summary ../artifacts/formal/summary.json \
  --server-log ../artifacts/server.log \
  --output-dir ../artifacts/report \
  --qps-window-s 1
```

This writes `report.md` and `report.json` atomically. The command uses only the
Python standard library.

Metric definitions:

- TTFT and TPOT are per-request streaming metrics with linear percentile
  interpolation.
- Per-user decode TPS is
  `(completion_tokens - 1) / (latency - TTFT)`, followed by P50 across users.
- Concurrency is an exact time-weighted distribution over inferred request
  start/end intervals, including idle gaps.
- QPS uses request arrivals in fixed non-overlapping windows, including empty
  windows. `--qps-window-s` defaults to one second.
- Acceptance P50/P99 use server-log samples inside the formal replay interval;
  mean acceptance length uses the final cumulative value in `summary.json`.
- Cache hit rate is `cached_tokens / prompt_tokens` per successful request.
- ISL and OSL histograms use fixed power-of-two token ranges and include count,
  percentage, cumulative percentage, and an ASCII bar.
- Token-level inter-token latency is intentionally not collected or reported.

## Validate the package

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile \
  scripts/replay_trace.py scripts/report_replay.py \
  tests/test_replay_trace.py tests/test_report_replay.py
bash -n scripts/launch_sglang.sh \
  scripts/run_smoke_replay.sh scripts/run_full_replay.sh
```

## Artifact policy

The original trace and raw result files may contain workload content and are
large enough to make the branch cumbersome. They are deliberately excluded.
Only the aggregate report and reproducible tooling are committed.
