#!/bin/bash
# Remove the per-job uv venv created by ci_install_dependency.sh.
#
# Meant to run in a post-job workflow step with `if: always()` so the venv is
# destroyed even on job failure/cancel. Runner-level safety net: a cron or
# startup task should also purge stale /tmp/sglang-ci-* directories to catch
# cancelled or crashed jobs that never reached this cleanup.

# Best-effort cleanup: never fail the job.
set +e
set -u

# Skip entirely when venv mode is disabled — no /tmp/sglang-ci-* dir exists
# and there's nothing to sweep. Matches the USE_VENV parsing in
# ci_install_dependency.sh (accepts 1/true/yes, case-insensitive).
USE_VENV_RAW="${USE_VENV:-true}"
case "$(printf '%s' "$USE_VENV_RAW" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes) ;;
    *)
        echo "USE_VENV=${USE_VENV_RAW}: skipping venv cleanup"
        exit 0
        ;;
esac

# Prefer the path propagated via GITHUB_ENV. Fallback: the stable path used by
# ci_install_dependency.sh (covers the case where install crashed before
# exporting SGLANG_CI_VENV_PATH).
STABLE_VENV="/tmp/sglang-ci-venv"
TARGET="${SGLANG_CI_VENV_PATH:-$STABLE_VENV}"
if [ -d "$TARGET" ]; then
    if rm -rf "$TARGET"; then
        echo "Cleaned up venv: $TARGET"
    else
        echo "::warning::Failed to remove $TARGET — runner cron should sweep /tmp/sglang-ci-*"
    fi
else
    echo "No venv to clean at $TARGET"
fi

# Sweep stale venvs from cancelled/crashed jobs that never reached cleanup.
# Any /tmp/sglang-ci-* dir older than 4 hours is considered orphaned.
stale_count=0
for venv in /tmp/sglang-ci-*; do
    [ -d "$venv" ] || continue
    if find "$venv" -maxdepth 0 -mmin +240 -print -quit | grep -q .; then
        rm -rf "$venv" && stale_count=$((stale_count + 1))
    fi
done
[ "$stale_count" -gt 0 ] && echo "Swept $stale_count stale venv(s) older than 4h"

exit 0
