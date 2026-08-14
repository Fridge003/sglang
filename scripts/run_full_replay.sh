#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TRACE_ZIP OUTPUT_DIR" >&2
  exit 2
fi

trace_zip=$1
output_dir=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python3 "$script_dir/replay_trace.py" \
  --trace "$trace_zip" \
  --base-url http://localhost:30000/v1 \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --output-dir "$output_dir" \
  --time-scale 1.0 \
  --request-timeout 7200 \
  --progress-every 25 \
  --flush-before-run \
  --reset-spec-before-run \
  --fail-fast \
  --force \
  2>&1 | tee "$output_dir.console.log"
