# DeepSeek-V4-Flash-0731 B200 max-running-8 / chunk-8192 trajectory report

Generated: `2026-08-15T10:38:59.052775Z`

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

The server was launched with the requested command, including cache reporting,
eight running requests, and an 8,192-token chunked-prefill limit.

```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tp 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --disable-flashinfer-autotune \
  --swa-full-tokens-ratio 0.1 \
  --mem-fraction-static 0.85 \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 30000 \
  --max-running-requests 8 \
  --chunked-prefill-size 8192
```

The saved `/server_info` snapshot verifies TP8, DSPARK,
`flashinfer_mxfp4`, `mem_fraction_static=0.85`, cache reporting,
`max_running_requests=8`, and `chunked_prefill_size=8192`. Because this
command does not set `--speculative-dspark-block-size`, the resolved block-size
field is `null` and SGLang resolves `speculative_num_draft_tokens=6`.

## Run summary

- Workload: 267 trajectories / 8,020 requests
- Requests: 8,020 successful / 0 failed / 8,020 total
- Prompt tokens: 180,742,867
- Completion tokens: 12,413,261
- Replay duration: 22,445.139023 seconds
- Original arrival span: 7,712.167252 seconds
- QPS window: 1.000000 seconds

The replay client preserves the trace's original arrival schedule. Concurrency
and TTFT include client-visible server queueing under the eight-running-request
admission limit.

## Central metrics

| Metric | Value | Unit |
| --- | ---: | --- |
| P50 TTFT | 130,408.431918 | ms |
| P99 TTFT | 529,864.430050 | ms |
| P50 TPOT | 10.771235 | ms/token |
| P99 TPOT | 19.435721 | ms/token |
| P50 Decode TPS per user | 92.839869 | token/s |
| Aggregate Decode TPS | 553.048969 | token/s |
| P50 concurrency | 66.000000 | requests |
| P99 concurrency | 152.000000 | requests |
| Mean concurrency | 71.448130 | requests |
| P50 QPS | 0.000000 | requests/s |
| P99 QPS | 3.000000 | requests/s |
| P50 cache hit rate | 95.800416 | % |
| P99 cache hit rate | 99.647899 | % |

## Acceptance Length

| Metric | Value | Source |
| --- | ---: | --- |
| P50 | 3.580000 | in-window server-log samples |
| P99 | 5.280000 | in-window server-log samples |
| Mean | 3.629616 | summary |
| Log sample mean | 3.577445 | 14,464 samples |

## Comparison with the max-running-4 run

The reference is `b200-latest-main-trajectory-20260814.json`. Lower is better
for duration, TTFT, and TPOT; higher is better for TPS and acceptance length.

| Metric | max-running-4 | max-running-8 / chunk-8192 | Change |
| --- | ---: | ---: | ---: |
| Replay duration (s) | 25,543.646511 | 22,445.139023 | -12.13% |
| P50 TTFT (ms) | 216,191.950320 | 130,408.431918 | -39.68% |
| P99 TTFT (ms) | 885,000.190425 | 529,864.430050 | -40.13% |
| P50 TPOT (ms/token) | 7.314170 | 10.771235 | +47.27% |
| P99 TPOT (ms/token) | 15.168121 | 19.435721 | +28.14% |
| P50 Decode TPS/user | 136.720914 | 92.839869 | -32.10% |
| Aggregate Decode TPS | 485.962762 | 553.048969 | +13.80% |
| P50 concurrency | 90 | 66 | -26.67% |
| P99 concurrency | 179 | 152 | -15.08% |
| Mean concurrency | 91.293770 | 71.448130 | -21.74% |
| P99 QPS | 2 | 3 | +1 request/s |
| P50 cache hit rate | 95.804703% | 95.800416% | -0.004287 pp |
| P99 cache hit rate | 99.647490% | 99.647899% | +0.000409 pp |
| Acceptance P50 | 3.380000 | 3.580000 | +5.92% |
| Acceptance P99 | 4.348000 | 5.280000 | +21.44% |
| Acceptance mean | 3.405306 | 3.629616 | +6.59% |

The new configuration improves aggregate decode throughput by 13.80%, shortens
the full replay by 12.13%, and cuts median and tail TTFT by about 40%. The
tradeoff is material: median per-user decode throughput falls 32.10%, with P50
TPOT 47.27% higher. The lower observed concurrency percentiles mean the faster
aggregate service drains the client backlog sooner; they do not mean the
server's admission limit was ignored.

This is not a single-variable concurrency comparison. Relative to the reference,
the new run changes `max_running_requests` from 4 to 8, changes the resolved
chunked-prefill size from 16,384 to 8,192, and removes the explicit DSPARK block
size of 4, resolving to six draft tokens. The deltas therefore describe the
requested configuration as a whole and must not be attributed to one flag.

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

- requests: `/scratch/baizhou/dsv4_replay_maxr8_chunk8192_20260814/replay/formal/requests.jsonl`
- server log: `/scratch/baizhou/dsv4_replay_maxr8_chunk8192_20260814/logs/server.log`
- summary: `/scratch/baizhou/dsv4_replay_maxr8_chunk8192_20260814/replay/formal/summary.json`
- `/server_info`: `/scratch/baizhou/dsv4_replay_maxr8_chunk8192_20260814/logs/server_info_before_replay.json`
- aggregate JSON: `b200-maxr8-chunk8192-trajectory-20260814.json`
- local archive: `/Users/baizhou.zhang/Desktop/dsv4 optimize/dsv4-replay-maxr8-chunk8192-20260814`

The raw request JSONL, generated text, and server log are retained in the local
experiment archive but are not committed because they may contain workload
content.

## Definitions

- Per-user decode TPS is `(completion_tokens - 1) / (latency - TTFT)`.
- Aggregate decode TPS is total completion tokens divided by replay duration.
- Concurrency percentiles are time-weighted over the full inferred replay interval.
- QPS is based on request arrivals in fixed non-overlapping windows, including empty windows.
- Cache hit rate is `cached_tokens / prompt_tokens` per successful request.
- Ordinary percentiles use linear interpolation at `(N - 1) * p / 100`.
