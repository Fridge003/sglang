#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 REMOTE_ARTIFACT_DIR" >&2
  exit 2
fi

artifact_dir=$1
mkdir -p "$artifact_dir"

nohup setsid sglang serve \
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
  >"$artifact_dir/server.log" 2>&1 < /dev/null &

server_pid=$!
echo "$server_pid" >"$artifact_dir/server.pid"
echo "Started SGLang PID $server_pid; log: $artifact_dir/server.log"
