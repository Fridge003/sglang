#!/bin/bash
# Install the main sglang package, skipping dependency resolution when
# python/pyproject.toml is unchanged since the last successful full install.
#
# Expected environment (set by ci_install_dependency.sh):
#   REPO_ROOT, PIP_CMD, PIP_INSTALL_SUFFIX, CU_VERSION, OPTIONAL_DEPS,
#   CUSTOM_BUILD_SGL_KERNEL (optional)
#
# Usage: source scripts/ci/cuda/ci_install_main_package.sh
set -euxo pipefail

EXTRAS="dev"
if [ -n "${OPTIONAL_DEPS:-}" ]; then
    EXTRAS="dev,${OPTIONAL_DEPS}"
fi
echo "Installing python extras: [${EXTRAS}]"

source "$(dirname "${BASH_SOURCE[0]}")/cache_nvidia_wheels.sh"

# Hash-based skip: if pyproject.toml hasn't changed, skip dependency resolution
# (--no-deps) to save ~30-100s. All other steps in ci_install_dependency.sh
# (apt, flashinfer, sglang-kernel, fix deps, etc.) still run unconditionally.
PYPROJECT_HASH=$(sha256sum "${REPO_ROOT}/python/pyproject.toml" | awk '{print $1}')
WORKSPACE_KEY=$(echo "${GITHUB_WORKSPACE:-$REPO_ROOT}" | md5sum | cut -c1-8)
HASH_FILE="/root/.cache/sglang-deps-hash-${WORKSPACE_KEY}-${OPTIONAL_DEPS:-base}"
SKIP_DEPS=false

if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ]; then
    echo "CUSTOM_BUILD_SGL_KERNEL=true, running full install"
elif [ -f "$HASH_FILE" ] && [ "$(cat "$HASH_FILE")" = "$PYPROJECT_HASH" ]; then
    echo "pyproject.toml unchanged (hash: ${PYPROJECT_HASH}), skipping dependency resolution"
    SKIP_DEPS=true
else
    echo "pyproject.toml changed or no cached hash, running full install"
fi

if [ "$SKIP_DEPS" = true ]; then
    $PIP_CMD install -e "python[${EXTRAS}]" --no-deps $PIP_INSTALL_SUFFIX
else
    $PIP_CMD install -e "python[${EXTRAS}]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION} $PIP_INSTALL_SUFFIX
    # Write hash only after successful install (set -e ensures we don't get here on failure)
    mkdir -p "$(dirname "$HASH_FILE")"
    echo "$PYPROJECT_HASH" > "$HASH_FILE"
    echo "Stored pyproject.toml hash: $PYPROJECT_HASH -> $HASH_FILE"
fi
