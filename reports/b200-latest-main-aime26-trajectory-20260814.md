# DeepSeek-V4-Flash-0731 B200 AIME26 and trajectory report

Generated: `2026-08-14T23:48:11.677361Z`

## Environment

- Devbox: `baizhou-dev-b200`
- GPU: 8 x NVIDIA B200
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- SGLang commit: `18107e38d266a98f59a8e3c766f8ddaf9b723ded`
- SGLang package: `0.0.0.dev16590+g18107e38d`
- Torch: `2.13.0`
- SGLang kernel: `0.4.6.post1`
- FlashInfer Python/cubin/JIT cache: `0.6.17` / `0.6.17` /
  `0.6.17+cu130`

## Server setting

The requested `--max-concurreny 4` setting maps to SGLang's
`--max-running-requests 4`. `--enable-cache-report` is observation-only and is
required for per-request cache-hit statistics.

```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tp 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 4 \
  --disable-flashinfer-autotune \
  --swa-full-tokens-ratio 0.1 \
  --mem-fraction-static 0.85 \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 30000 \
  --max-running-requests 4
```

## AIME26

`sgl-eval` used the `deepseek-ai/DeepSeek-V4-Flash-0731` model preset with 16
repeats, temperature 1.0, top-p 0.95, `max_tokens=200000`, thinking enabled,
and reasoning effort `max`.

| Metric | Result |
| --- | ---: |
| Requests | 480 / 480 completed |
| pass@1, average of 16 repeats | 98.3333% |
| pass@1 standard deviation | 2.7217% |
| pass@1 standard error | 0.6804% |
| pass@16 | 100.0000% |
| majority@16 | 100.0000% |
| Stop rate | 99.7917% |
| Truncated rate | 0.2083% |
| Error rate | 0.0000% |
| Output throughput | 537.3181 token/s |

The 16 output files contain 480 unique records: 472 are symbolically correct,
479 finished with `stop`, and one finished at the length limit. The exact
`sgl-eval` metrics are checked in as
`b200-latest-main-aime26-20260814.json`.

## Run summary

- Requests: 8,020 successful / 0 failed / 8,020 total
- Prompt tokens: 180,742,867
- Completion tokens: 12,413,261
- Replay duration: 25,543.646511 seconds
- QPS window: 1.000000 seconds

The replay client preserves the trace's original arrival schedule while the
server admits at most four running requests. TTFT and concurrency therefore
include server-side queueing under this requested limit.

## Central metrics

| Metric | Value | Unit |
| --- | ---: | --- |
| P50 TTFT | 216,191.950320 | ms |
| P99 TTFT | 885,000.190425 | ms |
| P50 TPOT | 7.314170 | ms/token |
| P99 TPOT | 15.168121 | ms/token |
| P50 Decode TPS per user | 136.720914 | token/s |
| P50 concurrency | 90.000000 | requests |
| P99 concurrency | 179.000000 | requests |
| P50 QPS | 0.000000 | requests/s |
| P99 QPS | 2.000000 | requests/s |
| P50 cache hit rate | 95.804703 | % |
| P99 cache hit rate | 99.647490 | % |

## Acceptance Length

| Metric | Value | Source |
| --- | ---: | --- |
| P50 | 3.380000 | in-window server-log samples |
| P99 | 4.348000 | in-window server-log samples |
| Mean | 3.405306 | summary |
| Log sample mean | 3.401157 | 23,621 samples |

## ISL Histogram

| Range | Count | Percentage | Cumulative | Histogram |
| --- | ---: | ---: | ---: | --- |
| 0-128 | 0 | 0.0000% | 0.0000% | `` |
| 129-256 | 0 | 0.0000% | 0.0000% | `` |
| 257-512 | 0 | 0.0000% | 0.0000% | `` |
| 513-1024 | 0 | 0.0000% | 0.0000% | `` |
| 1025-2048 | 185 | 2.3067% | 2.3067% | `###` |
| 2049-4096 | 1,068 | 13.3167% | 15.6234% | `#################` |
| 4097-8192 | 1,512 | 18.8529% | 34.4763% | `#########################` |
| 8193-16384 | 1,840 | 22.9426% | 57.4190% | `##############################` |
| 16385-32768 | 1,667 | 20.7855% | 78.2045% | `###########################` |
| 32769-65536 | 1,185 | 14.7756% | 92.9800% | `###################` |
| 65537-131072 | 526 | 6.5586% | 99.5387% | `#########` |
| >131072 | 37 | 0.4613% | 100.0000% | `#` |

## OSL Histogram

| Range | Count | Percentage | Cumulative | Histogram |
| --- | ---: | ---: | ---: | --- |
| 0-128 | 2,302 | 28.7032% | 28.7032% | `##############################` |
| 129-256 | 1,832 | 22.8429% | 51.5461% | `########################` |
| 257-512 | 1,310 | 16.3342% | 67.8803% | `#################` |
| 513-1024 | 897 | 11.1845% | 79.0648% | `############` |
| 1025-2048 | 630 | 7.8554% | 86.9202% | `########` |
| 2049-4096 | 456 | 5.6858% | 92.6060% | `######` |
| 4097-8192 | 296 | 3.6908% | 96.2968% | `####` |
| 8193-16384 | 147 | 1.8329% | 98.1297% | `##` |
| 16385-32768 | 97 | 1.2095% | 99.3392% | `#` |
| 32769-65536 | 53 | 0.6608% | 100.0000% | `#` |
| 65537-131072 | 0 | 0.0000% | 100.0000% | `` |
| >131072 | 0 | 0.0000% | 100.0000% | `` |

## Source artifacts

- requests: `/scratch/baizhou/dsv4_aime26_trajectory_20260814/replay/formal/requests.jsonl`
- server_log: `/scratch/baizhou/dsv4_aime26_trajectory_20260814/logs/server.log`
- summary: `/scratch/baizhou/dsv4_aime26_trajectory_20260814/replay/formal/summary.json`
- aggregate JSON: `b200-latest-main-trajectory-20260814.json`

The raw trace, request JSONL, generated text, and server log are retained in
the local experiment archive but are not committed because they are large and
may contain workload content.

## Definitions

- Per-user decode TPS is `(completion_tokens - 1) / (latency - TTFT)`.
- Concurrency percentiles are time-weighted over the full inferred replay interval.
- QPS is based on request arrivals in fixed non-overlapping windows, including empty windows.
- Cache hit rate is `cached_tokens / prompt_tokens` per successful request.
- Ordinary percentiles use linear interpolation at `(N - 1) * p / 100`.
