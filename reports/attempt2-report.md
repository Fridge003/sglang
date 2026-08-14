# DSV4 replay report

Generated: `2026-08-14T06:56:40.279385Z`

## Run summary

- Requests: 8,020 successful / 0 failed / 8,020 total
- Prompt tokens: 180,742,867
- Completion tokens: 12,413,261
- Replay duration: 19,334.645255 seconds
- QPS window: 1.000000 seconds

## Central metrics

| Metric | Value | Unit |
| --- | ---: | --- |
| P50 TTFT | 536.049035 | ms |
| P99 TTFT | 2,106.461418 | ms |
| P50 TPOT | 40.978758 | ms/token |
| P99 TPOT | 94.651233 | ms/token |
| P50 Decode TPS per user | 24.402887 | token/s |
| P50 concurrency | 21.000000 | requests |
| P99 concurrency | 65.000000 | requests |
| P50 QPS | 0.000000 | requests/s |
| P99 QPS | 3.000000 | requests/s |
| P50 cache hit rate | 95.804323 | % |
| P99 cache hit rate | 99.648266 | % |

## Acceptance Length

| Metric | Value | Source |
| --- | ---: | --- |
| P50 | 3.330000 | in-window server-log samples |
| P99 | 4.885200 | in-window server-log samples |
| Mean | 3.640244 | summary |
| Log sample mean | 3.314702 | 8,475 samples |

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

- requests: `../remote/attempt2/formal/requests.jsonl`
- server_log: `../remote/attempt2/server.log`
- summary: `../remote/attempt2/formal/summary.json`

## Definitions

- Per-user decode TPS is `(completion_tokens - 1) / (latency - TTFT)`.
- Concurrency percentiles are time-weighted over the full inferred replay interval.
- QPS is based on request arrivals in fixed non-overlapping windows, including empty windows.
- Cache hit rate is `cached_tokens / prompt_tokens` per successful request.
- Ordinary percentiles use linear interpolation at `(N - 1) * p / 100`.
