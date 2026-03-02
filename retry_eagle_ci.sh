#!/bin/bash
set -euo pipefail

export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")"

REPO="sgl-project/sglang"
BRANCH="${BRANCH:-$(git branch --show-current)}"
WORKFLOW="pr-test.yml"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
STAGES=("stage-b-test-large-1-gpu" "stage-b-test-small-1-gpu")

echo "Eagle CI Retry | branch=$BRANCH stages=${STAGES[*]} max=$MAX_ITERATIONS"

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "#$i  TRIGGER  $(date '+%H:%M:%S')"

    # Trigger both stages
    RUN_IDS=()
    for stage in "${STAGES[@]}"; do
        gh workflow run "$WORKFLOW" --repo "$REPO" --ref "$BRANCH" \
            -f target_stage="$stage" 2>/dev/null
        sleep 5
    done

    # Find run IDs
    sleep 15
    RUN_IDS=()
    for _ in "${STAGES[@]}"; do
        RUN_IDS=()
        RUNS=$(gh run list --repo "$REPO" --workflow "$WORKFLOW" \
            --branch "$BRANCH" --limit "${#STAGES[@]}" --json databaseId,status \
            -q '.[].databaseId' 2>/dev/null || true)
        while IFS= read -r rid; do
            [[ -n "$rid" ]] && RUN_IDS+=("$rid")
        done <<< "$RUNS"
        [[ ${#RUN_IDS[@]} -ge ${#STAGES[@]} ]] && break
        sleep 10
    done

    for idx in "${!RUN_IDS[@]}"; do
        echo "#$i  START   [${STAGES[$idx]:-?}] https://github.com/$REPO/actions/runs/${RUN_IDS[$idx]}  $(date '+%H:%M:%S')"
    done

    # Poll all runs until done
    CRASHED=false
    for rid in "${RUN_IDS[@]}"; do
        RUN_URL="https://github.com/$REPO/actions/runs/$rid"
        while true; do
            STATUS=$(gh run view "$rid" --repo "$REPO" --json status,conclusion \
                -q '[.status,.conclusion] | join(",")' 2>/dev/null || echo "unknown,")
            case "$STATUS" in
                completed,success)
                    echo "#$i  PASS    $RUN_URL  $(date '+%H:%M:%S')"
                    break
                    ;;
                completed,*)
                    CONCLUSION="${STATUS#completed,}"
                    echo "#$i  FAIL($CONCLUSION)  $RUN_URL  $(date '+%H:%M:%S')"
                    CRASHED=true
                    break
                    ;;
                *)
                    sleep 60
                    ;;
            esac
        done
    done

    if $CRASHED; then
        echo "CRASH DETECTED on iteration $i — $(date '+%Y-%m-%d %H:%M:%S')"
        exit 0
    fi

    sleep "$SLEEP_BETWEEN"
done

echo "No crash after $MAX_ITERATIONS iterations."
exit 1
