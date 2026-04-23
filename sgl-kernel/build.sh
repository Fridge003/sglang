#!/bin/bash
set -ex

if [ $# -lt 2 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION> [ARCH]"
  exit 1
fi

PYTHON_VERSION="$1"          # e.g. 3.10
CUDA_VERSION="$2"            # e.g. 12.9
ARCH="${3:-$(uname -i)}"     # optional override

if [ "${ARCH}" = "aarch64" ]; then
  BASE_IMG="pytorch/manylinuxaarch64-builder"
else
  BASE_IMG="pytorch/manylinux2_28-builder"
fi

# Create cache directories for persistent build artifacts in home directory
# Using home directory to persist across workspace cleanups/checkouts
CACHE_DIR="${HOME}/.cache/sgl-kernel"
BUILDX_CACHE_DIR="${CACHE_DIR}/buildx"
CCACHE_HOST_DIR="${CACHE_DIR}/ccache"
mkdir -p "${BUILDX_CACHE_DIR}" "${CCACHE_HOST_DIR}"

# Ensure a buildx builder with docker-container driver (required for cache export)
BUILDER_NAME="sgl-kernel-builder"
# RESET_BUILDER=1 removes and recreates the builder to clear corrupted internal
# state (e.g. stale containerd snapshots from base image layer GC).
if [ "${RESET_BUILDER:-0}" = "1" ]; then
  echo "Resetting buildx builder: ${BUILDER_NAME}"
  docker buildx rm "${BUILDER_NAME}" 2>/dev/null || true
  rm -rf "${BUILDX_CACHE_DIR}"
  mkdir -p "${BUILDX_CACHE_DIR}"
fi
if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use --bootstrap
else
  docker buildx use "${BUILDER_NAME}"
fi

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

# Output directory for wheels
DIST_DIR="dist"
mkdir -p "${DIST_DIR}"

echo "----------------------------------------"
echo "Build configuration"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "ARCH:           ${ARCH}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "PYTHON_TAG:     ${PY_TAG}"
echo "Output:         ${DIST_DIR}/"
echo "Buildx cache:   ${BUILDX_CACHE_DIR}"
echo "ccache dir:     ${CCACHE_HOST_DIR}"
echo "Builder:        ${BUILDER_NAME}"
echo "BUILD_JOBS:     ${BUILD_JOBS:-auto}"
echo "NVCC_THREADS:   ${NVCC_THREADS:-2}"
echo "USE_CCACHE:     ${USE_CCACHE:-1}"
echo "RESET_BUILDER:  ${RESET_BUILDER:-0}"
echo "----------------------------------------"

# Optional build-args (empty string disables)
BUILD_ARGS=()
[ -n "${ENABLE_CMAKE_PROFILE:-}" ] && BUILD_ARGS+=(--build-arg ENABLE_CMAKE_PROFILE="${ENABLE_CMAKE_PROFILE}")
[ -n "${ENABLE_BUILD_PROFILE:-}" ] && BUILD_ARGS+=(--build-arg ENABLE_BUILD_PROFILE="${ENABLE_BUILD_PROFILE}")
[ -n "${USE_CCACHE:-}" ]           && BUILD_ARGS+=(--build-arg USE_CCACHE="${USE_CCACHE}")
[ -n "${BUILD_JOBS:-}" ]           && BUILD_ARGS+=(--build-arg BUILD_JOBS="${BUILD_JOBS}")
[ -n "${NVCC_THREADS:-}" ]         && BUILD_ARGS+=(--build-arg NVCC_THREADS="${NVCC_THREADS}")

# ---- Step 1: Build deps image (layer cached, fast on repeat) ----
DEPS_TAG="sgl-kernel-deps:cuda${CUDA_VERSION}-${PY_TAG}-${ARCH}"

docker buildx build \
  --builder "${BUILDER_NAME}" \
  -f Dockerfile . \
  --build-arg BASE_IMG="${BASE_IMG}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg ARCH="${ARCH}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg PYTHON_TAG="${PY_TAG}" \
  "${BUILD_ARGS[@]}" \
  --cache-from "type=local,src=${BUILDX_CACHE_DIR}" \
  --cache-to "type=local,dest=${BUILDX_CACHE_DIR},mode=max" \
  --target deps \
  --load \
  -t "${DEPS_TAG}" \
  --network=host

echo "Deps image ready: ${DEPS_TAG}"

# ---- Step 2: Build wheel with host-mounted ccache ----
# This allows ccache to persist on the host filesystem across builds.
CCACHE_FLAG="${USE_CCACHE:-1}"
BUILD_JOBS_FLAG="${BUILD_JOBS:-0}"
NVCC_THREADS_FLAG="${NVCC_THREADS:-2}"

docker run --rm \
  --network=host \
  -v "$(pwd):/sgl-kernel" \
  -v "${CCACHE_HOST_DIR}:/ccache" \
  -w /sgl-kernel \
  -e ARCH="${ARCH}" \
  "${DEPS_TAG}" \
  bash -c '
set -eux

USE_CCACHE='"${CCACHE_FLAG}"'
BUILD_JOBS='"${BUILD_JOBS_FLAG}"'
NVCC_THREADS='"${NVCC_THREADS_FLAG}"'

if [ "${USE_CCACHE}" = "1" ]; then
  export CCACHE_DIR=/ccache
  export CCACHE_BASEDIR=/sgl-kernel
  export CCACHE_MAXSIZE=10G
  export CCACHE_COMPILERCHECK=content
  export CCACHE_COMPRESS=true
  export CCACHE_SLOPPINESS=file_macro,time_macros,include_file_mtime,include_file_ctime
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
  export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
  echo "=== ccache stats (before) ==="
  ccache -sV
fi

if [ "'"${ARCH}"'" = "aarch64" ]; then
  export CUDA_NVCC_FLAGS="-Xcudafe --threads=8"
  export MAKEFLAGS="-j8"
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  export NINJAFLAGS="-j4"
  # Hand-tuned aarch64 path predates the cmake-side auto pool; turn the pool
  # off so it does not further clamp these conservative settings to 1.
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DSGL_KERNEL_MINIMIZE_BUILD_MEMORY=OFF"
  echo "ARM detected: CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}, MAKEFLAGS=${MAKEFLAGS}, NINJAFLAGS=${NINJAFLAGS}"
elif [ "${BUILD_JOBS}" -gt 0 ] 2>/dev/null; then
  export CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
else
  # Default: clamp by both CPU and available RAM. CUTLASS/flashinfer TUs peak
  # around ~5 GB each, so big-CPU/modest-RAM runners OOM without a mem clamp.
  # The cmake JOB_POOLS in CMakeLists.txt do the precise per-stage budgeting;
  # this block is a coarser ninja -j ceiling for callers who skip the pool.
  CPU_JOBS=$(( $(nproc) * 2 / 3 ))
  MEM_KB=$(awk "/MemAvailable/ {print \$2}" /proc/meminfo)
  if [ -z "${MEM_KB}" ]; then
    echo "WARNING: MemAvailable not present in /proc/meminfo; falling back to CPU-only sizing" >&2
    MEM_BOUND=${CPU_JOBS}
  else
    MEM_BOUND=$(( MEM_KB / 1024 / 1024 / 5 ))
    [ "${MEM_BOUND}" -lt 1 ] && MEM_BOUND=1
  fi
  JOBS=${CPU_JOBS}
  [ "${MEM_BOUND}" -lt "${JOBS}" ] && JOBS=${MEM_BOUND}
  [ "${JOBS}" -gt 64 ] && JOBS=64
  echo "Auto parallelism: CPU_JOBS=${CPU_JOBS}, MEM_KB=${MEM_KB:-unknown}, MEM_BOUND=${MEM_BOUND} -> JOBS=${JOBS}"
  export CMAKE_BUILD_PARALLEL_LEVEL=${JOBS}
fi

export CMAKE_ARGS="${CMAKE_ARGS:-} -DSGL_KERNEL_COMPILE_THREADS=${NVCC_THREADS}"
echo "Build parallelism: CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}, NVCC_THREADS=${NVCC_THREADS}"

${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation
PYTHON=${PYTHON_ROOT_PATH}/bin/python ./rename_wheels.sh

if [ "${USE_CCACHE}" = "1" ]; then
  echo "=== ccache stats (after) ==="
  ccache -s
fi
'

echo "Done. Wheels are in ${DIST_DIR}/"
ls -lh "${DIST_DIR}"/*.whl 2>/dev/null || true
