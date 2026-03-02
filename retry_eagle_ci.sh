#!/bin/bash
set -euo pipefail

REPO="sgl-project/sglang"
BRANCH="${BRANCH:-$(git branch --show-current)}"
WORKFLOW="pr-test.yml"
TARGET_STAGE="${TARGET_STAGE:-stage-b-test-large-1-gpu}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"

echo "========================================"
echo "Eagle CI Retry — until crash"
echo "  Repo:         $REPO"
echo "  Branch:       $BRANCH"
echo "  Workflow:     $WORKFLOW"
echo "  Target stage: $TARGET_STAGE"
echo "  Max iters:    $MAX_ITERATIONS"
echo "========================================"

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo ""
    echo "=== Iteration $i / $MAX_ITERATIONS  $(date '+%Y-%m-%d %H:%M:%S') ==="

    # 1. Trigger workflow
    echo "Triggering workflow..."
    gh workflow run "$WORKFLOW" --repo "$REPO" --ref "$BRANCH" \
        -f target_stage="$TARGET_STAGE"

    # 2. Wait for the run to appear
    sleep 15
    RUN_ID=""
    for attempt in $(seq 1 12); do
        RUN_ID=$(gh run list --repo "$REPO" --workflow "$WORKFLOW" \
            --branch "$BRANCH" --limit 1 --json databaseId,status \
            -q '.[0].databaseId' 2>/dev/null || true)
        if [[ -n "$RUN_ID" ]]; then
            break
        fi
        echo "  Waiting for run to appear (attempt $attempt)..."
        sleep 10
    done

    if [[ -z "$RUN_ID" ]]; then
        echo "ERROR: Could not find run after trigger. Retrying..."
        sleep "$SLEEP_BETWEEN"
        continue
    fi

    RUN_URL="https://github.com/$REPO/actions/runs/$RUN_ID"
    echo "Run $RUN_ID: $RUN_URL"

    # 3. Watch until completion
    echo "Watching run..."
    if gh run watch "$RUN_ID" --repo "$REPO" --exit-status 2>&1; then
        echo "PASS on iteration $i. Sleeping ${SLEEP_BETWEEN}s before retry..."
        sleep "$SLEEP_BETWEEN"
    else
        echo ""
        echo "========================================"
        echo "CRASH DETECTED on iteration $i!"
        echo "  Run: $RUN_URL"
        echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"
        echo ""
        echo "View logs:  gh run view $RUN_ID --repo $REPO --log"
        echo "View jobs:  gh run view $RUN_ID --repo $REPO"
        exit 0
    fi
done

echo ""
echo "No crash after $MAX_ITERATIONS iterations."
exit 1
