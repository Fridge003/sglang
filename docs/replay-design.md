# DeepSeek-V4 TerminalBench Trace Replay Design

## Goal

Replay `session__e3660f54.zip` against an SGLang DeepSeek-V4-Flash-0731
server on `baizhou-dev-b200` and collect request-level and aggregate serving
performance metrics.

## Trace scope

- 267 agent trajectories and 8,020 recorded model invocations.
- 180,872,096 recorded prompt tokens and 12,413,260 recorded completion tokens.
- Original rollout window: 10,430.665 seconds.
- Each invocation is reconstructed from the recorded event prefix. Generated
  output from replay is measured but is not substituted into later historical
  requests.

## Replay semantics

Each trajectory is an independent asynchronous worker. Workers start at their
recorded offset from the earliest trajectory start. Within a worker, model
invocations are sent sequentially. After a recorded tool-call response, the
worker sleeps for the sum of the associated recorded tool-result durations
before sending the next invocation.

System and user messages retain their recorded text. Historical assistant
messages retain `content`, `reasoning_content`, and OpenAI-compatible tool
calls. Tool results are associated with the matching recorded tool-call IDs.
The trace's tool schemas are converted from `input_schema` to OpenAI function
schemas. The current invocation is represented only through `max_tokens`; its
recorded answer is never included in its own prompt.

Formal replay uses the recorded completion-token count as `max_tokens`. Failed
requests are recorded and do not stop unrelated trajectories.

## Server lifecycle

Launch the latest-main SGLang checkout on all eight B200 GPUs with the approved
command and port 30000. After readiness, run a short compatibility smoke test.
Then wait for idleness, flush the radix cache, and reset speculative decoding
counters through `/set_internal_state` with an empty server-argument update.
Only the subsequent full replay contributes to reported metrics.

## Metrics

- TTFT: request start to first non-empty streamed content, reasoning, or tool
  call delta.
- TPOT: `(request latency - TTFT) / (completion_tokens - 1)` for responses with
  at least two completion tokens.
- Decode TPS: total successful completion tokens divided by formal replay wall
  time.
- Cache-hit rate: per-request device cached tokens divided by actual prompt
  tokens; report P50, P99, and token-weighted aggregate.
- Mean acceptance length: `avg_spec_accept_length` read from `/server_info`
  after the reset and formal replay.

Percentiles use linear interpolation over successful requests. The report also
includes success/error counts, replay duration, and actual token totals.

## Artifacts

The local experiment directory contains the replay script, unit tests, server
launch command, remote logs, per-request JSONL, summary JSON, and a concise
Markdown report. Remote artifacts are copied back before the server is stopped.
