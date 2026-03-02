#!/bin/bash
set -euo pipefail

export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")"

REPO="sgl-project/sglang"
BRANCH="${BRANCH:-$(git branch --show-current)}"
WORKFLOW="pr-test.yml"
TARGET_STAGE="${TARGET_STAGE:-stage-b-test-large-1-gpu}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"

echo "Eagle CI Retry | branch=$BRANCH stage=$TARGET_STAGE max=$MAX_ITERATIONS"

for i in $(seq 1 "$MAX_ITERATIONS"); do
    # 1. Trigger
    gh workflow run "$WORKFLOW" --repo "$REPO" --ref "$BRANCH" \
        -f target_stage="$TARGET_STAGE" 2>/dev/null

    # 2. Find run ID
    sleep 15
    RUN_ID=""
    for _ in $(seq 1 12); do
        RUN_ID=$(gh run list --repo "$REPO" --workflow "$WORKFLOW" \
            --branch "$BRANCH" --limit 1 --json databaseId \
            -q '.[0].databaseId' 2>/dev/null || true)
        [[ -n "$RUN_ID" ]] && break
        sleep 10
    done
    [[ -z "$RUN_ID" ]] && echo "#$i  ERROR: run not found" && sleep "$SLEEP_BETWEEN" && continue

    RUN_URL="https://github.com/$REPO/actions/runs/$RUN_ID"

    # 3. Poll until done (silent)
    while true; do
        STATUS=$(gh run view "$RUN_ID" --repo "$REPO" --json status,conclusion \
            -q '[.status,.conclusion] | join(",")' 2>/dev/null || echo "unknown,")
        case "$STATUS" in
            completed,success)
                echo "#$i  PASS  $RUN_URL  $(date '+%H:%M:%S')"
                break
                ;;
            completed,*)
                CONCLUSION="${STATUS#completed,}"
                echo "#$i  FAIL($CONCLUSION)  $RUN_URL  $(date '+%H:%M:%S')"
                echo "CRASH DETECTED on iteration $i — $(date '+%Y-%m-%d %H:%M:%S')"
                exit 0
                ;;
            *)
                sleep 60
                ;;
        esac
    done

    sleep "$SLEEP_BETWEEN"
done

echo "No crash after $MAX_ITERATIONS iterations."
exit 1
