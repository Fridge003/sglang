#!/bin/bash
set -euo pipefail
HOSTNAME_VALUE=$(hostname)
GPU_ARCH="mi30x"   # default
SKIP_TT_DEPS=""
SKIP_SGLANG_BUILD=""
SKIP_AITER_BUILD=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-aiter-build) SKIP_AITER_BUILD="1"; shift;;
    --skip-sglang-build) SKIP_SGLANG_BUILD="1"; shift;;
    --skip-test-time-deps) SKIP_TT_DEPS="1"; shift;;
    -h|--help)
      echo "Usage: $0 [OPTIONS] [OPTIONAL_DEPS]"
      echo "Options:"
      echo "  --skip-sglang-build         Don't build checkout sglang, use what was shipped with the image"
      echo "  --skip-aiter-build          Don't build aiter, use what was shipped with the image"
      echo "  --skip-test-time-deps       Don't build miscellaneous dependencies"
      exit 0
      ;;
    *) break ;;
  esac
done

OPTIONAL_DEPS="${1:-}"

# Build python extras
EXTRAS="dev_hip"
if [ -n "$OPTIONAL_DEPS" ]; then
    EXTRAS="dev_hip,${OPTIONAL_DEPS}"
fi
echo "Installing python extras: [${EXTRAS}]"

# Host names look like: linux-mi35x-gpu-1-xxxxx-runner-zzzzz
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from hostname: ${GPU_ARCH}"
else
  echo "Warning: could not parse GPU architecture from '${HOSTNAME_VALUE}', defaulting to ${GPU_ARCH}"
fi

# Install the required dependencies in CI.
# Fix permissions on pip cache, ignore errors from concurrent access or missing temp files
docker exec ci_sglang chown -R root:root /sgl-data/pip-cache 2>/dev/null || true
docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache --upgrade pip

# Helper function to install with retries and fallback PyPI mirror
install_with_retry() {
  local max_attempts=3
  local cmd="$@"

  for attempt in $(seq 1 $max_attempts); do
    echo "Attempt $attempt/$max_attempts: $cmd"
    if eval "$cmd"; then
      echo "Success!"
      return 0
    fi

    if [ $attempt -lt $max_attempts ]; then
      echo "Failed, retrying in 5 seconds..."
      sleep 5
      # Try with alternative PyPI index on retry
      if [[ "$cmd" =~ "pip install" ]] && [ $attempt -eq 2 ]; then
        cmd="$cmd --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
        echo "Using fallback PyPI mirror: $cmd"
      fi
    fi
  done

  echo "Failed after $max_attempts attempts"
  return 1
}

# ============================================================================
# DEBUG INSTRUMENTATION (bingxche/fix-install-depend-error)
# Diagnostic install of `pip install -e python[$EXTRAS]` to capture exactly
# which package(s) are causing pip's resolver to explode with
# `resolution-too-deep`. Writes the full -vvv log to /tmp on the runner so
# the workflow can upload it as an artifact, and also prints a compact digest
# to the GitHub Actions log for quick triage.
# ============================================================================
install_sglang_with_debug() {
  local extras="$1"
  local ts log_dir log_attempt1 log_attempt2 log_attempt3
  ts=$(date +%Y%m%d-%H%M%S)
  log_dir="${RUNNER_TEMP:-/tmp}/pip-install-debug"
  mkdir -p "$log_dir"
  log_attempt1="${log_dir}/attempt1-vvv-${ts}.log"
  log_attempt2="${log_dir}/attempt2-legacy-resolver-${ts}.log"
  log_attempt3="${log_dir}/attempt3-aliyun-${ts}.log"

  echo "::group::[DEBUG] Pre-install environment baseline"
  echo "----- pip version inside container -----"
  docker exec ci_sglang pip --version || true
  docker exec ci_sglang python -V || true
  echo "----- pip config -----"
  docker exec ci_sglang pip config list || true
  echo "----- Pre-installed versions of dependencies most likely to backtrack -----"
  docker exec ci_sglang pip list --format=freeze 2>/dev/null | \
    grep -iE "^(transformers|tokenizers|huggingface[-_]hub|setproctitle|urllib3|certifi|requests|cache-dit|peft|diffusers|torch|easydict|aiohttp|accelerate|outlines|xgrammar|openai|mistral[-_]common|numpy|pillow|safetensors|llguidance|fsspec|regex)=" || true
  echo "----- Disk usage of container (root) -----"
  docker exec ci_sglang df -h / 2>/dev/null | head -3 || true
  echo "::endgroup::"

  echo "::group::[DEBUG] Attempt 1: pip install -vvv (full resolver trace -> ${log_attempt1})"
  set +e
  docker exec ci_sglang pip install -vvv --no-input --no-color \
    --cache-dir=/sgl-data/pip-cache -e "python[${extras}]" \
    > "$log_attempt1" 2>&1
  local rc1=$?
  set -e
  echo "Attempt 1 exit code: $rc1"
  echo "Attempt 1 log size: $(wc -c < "$log_attempt1" 2>/dev/null || echo 0) bytes, $(wc -l < "$log_attempt1" 2>/dev/null || echo 0) lines"
  echo "::endgroup::"

  if [ $rc1 -eq 0 ]; then
    echo "Attempt 1 succeeded — no debug needed."
    return 0
  fi

  echo "::group::[DEBUG] Attempt 1 failed — resolver log digest"
  echo "----- Last 5 'error/ERROR' lines -----"
  grep -E "^(error|ERROR)[: ]" "$log_attempt1" | tail -5 || true
  echo "----- 'resolution-too-deep' / 'taking longer' indicators -----"
  grep -nE "resolution-too-deep|taking longer than usual|backtracking" "$log_attempt1" | head -10 || true
  echo "----- Top 30 packages by 'Collecting' frequency (high = pip kept re-collecting it) -----"
  grep -oE "^Collecting [a-zA-Z0-9._-]+" "$log_attempt1" \
    | awk '{print $2}' | sort | uniq -c | sort -rn | head -30 || true
  echo "----- Top 30 packages by 'Using cached' frequency (high = many versions explored == backtrack hot spot) -----"
  grep -oE "Using cached [a-zA-Z0-9._-]+-[0-9]" "$log_attempt1" \
    | awk '{print $3}' | sed -E 's/-[0-9].*//' | sort | uniq -c | sort -rn | head -30 || true
  echo "----- Top 30 distinct (package, version) pairs pip downloaded metadata for -----"
  grep -oE "Downloading [a-zA-Z0-9._-]+-[0-9][a-zA-Z0-9.+_-]*" "$log_attempt1" \
    | awk '{print $2}' | sort | uniq -c | sort -rn | head -30 || true
  echo "----- Last 80 lines of attempt-1 log -----"
  tail -80 "$log_attempt1" || true
  echo "::endgroup::"

  echo "::group::[DEBUG] Attempt 2: pip install --use-deprecated=legacy-resolver (does the OLD resolver succeed?)"
  set +e
  docker exec ci_sglang pip install --no-input --no-color \
    --cache-dir=/sgl-data/pip-cache \
    --use-deprecated=legacy-resolver \
    -e "python[${extras}]" \
    > "$log_attempt2" 2>&1
  local rc2=$?
  set -e
  echo "Attempt 2 exit code: $rc2"
  echo "Attempt 2 log size: $(wc -c < "$log_attempt2" 2>/dev/null || echo 0) bytes"
  echo "----- Last 60 lines of attempt-2 log -----"
  tail -60 "$log_attempt2" || true
  echo "::endgroup::"

  if [ $rc2 -eq 0 ]; then
    echo "[DEBUG] Attempt 2 succeeded with legacy-resolver. This CONFIRMS the new resolver is the problem."
    return 0
  fi

  echo "::group::[DEBUG] Attempt 3: pip install with aliyun fallback mirror (last-resort, kept for parity with prod retry)"
  set +e
  docker exec ci_sglang pip install --no-input --no-color \
    --cache-dir=/sgl-data/pip-cache \
    --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    -e "python[${extras}]" \
    > "$log_attempt3" 2>&1
  local rc3=$?
  set -e
  echo "Attempt 3 exit code: $rc3"
  echo "----- Last 40 lines of attempt-3 log -----"
  tail -40 "$log_attempt3" || true
  echo "::endgroup::"

  if [ $rc3 -eq 0 ]; then
    return 0
  fi

  echo "[DEBUG] All 3 attempts failed. Full logs preserved under: ${log_dir}"
  ls -la "$log_dir" || true
  return 1
}

# Helper function to git clone with retries
git_clone_with_retry() {
  local repo_url="$1"
  local dest_dir="${2:-}"
  local branch_args="${3:-}"
  local max_attempts=3

  for attempt in $(seq 1 $max_attempts); do
    echo "Git clone attempt $attempt/$max_attempts: $repo_url"

    # prevent from partial clone
    if [ -n "$dest_dir" ] && [ -d "$dest_dir" ]; then
      rm -rf "$dest_dir"
    fi

    if git \
      -c http.lowSpeedLimit=1000 \
      -c http.lowSpeedTime=30 \
      clone --depth 1 ${branch_args:+$branch_args} "$repo_url" "$dest_dir"; then
      echo "Git clone succeeded."
      return 0
    fi

    if [ $attempt -lt $max_attempts ]; then
      echo "Git clone failed, retrying in 5 seconds..."
      sleep 5
    fi
  done

  echo "Git clone failed after $max_attempts attempts: $repo_url"
  return 1
}

# Install checkout sglang
if [ -n "$SKIP_SGLANG_BUILD" ]; then
  echo "Didn't build checkout SGLang"
else
  docker exec ci_sglang pip uninstall sgl-kernel -y || true
  docker exec ci_sglang pip uninstall sglang-kernel -y || true
  docker exec ci_sglang pip uninstall sglang -y || true
  # Clear Python cache to ensure latest code is used
  docker exec ci_sglang find /opt/venv -name "*.pyc" -delete || true
  docker exec ci_sglang find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + || true
  # Also clear cache in sglang-checkout
  docker exec ci_sglang find /sglang-checkout -name "*.pyc" -delete || true
  docker exec ci_sglang find /sglang-checkout -name "__pycache__" -type d -exec rm -rf {} + || true
  docker exec -w /sglang-checkout/sgl-kernel ci_sglang bash -c "rm -f pyproject.toml && mv pyproject_rocm.toml pyproject.toml && python3 setup_rocm.py install"

  docker exec ci_sglang bash -c 'rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml'
  install_sglang_with_debug "${EXTRAS}"
fi

if [[ -n "${SKIP_TT_DEPS}" ]]; then
  echo "Didn't build lmms_eval, human-eval, and others"
else
  # For lmms_evals evaluating MMMU
  docker exec -w / ci_sglang git clone --branch v0.4.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
  install_with_retry docker exec -w /lmms-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e .

  git_clone_with_retry https://github.com/akao-amd/human-eval.git human-eval
  docker cp human-eval ci_sglang:/
  install_with_retry docker exec -w /human-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e .

  docker exec -w / ci_sglang mkdir -p /dummy-grok
  # Create dummy grok config inline (bypasses Azure blob storage which may have auth issues)
  mkdir -p dummy-grok
  cat > dummy-grok/config.json << 'EOF'
  {
    "architectures": [
      "Grok1ModelForCausalLM"
    ],
    "embedding_multiplier_scale": 78.38367176906169,
    "output_multiplier_scale": 0.5773502691896257,
    "vocab_size": 131072,
    "hidden_size": 6144,
    "intermediate_size": 32768,
    "max_position_embeddings": 8192,
    "num_experts_per_tok": 2,
    "num_local_experts": 8,
    "num_attention_heads": 48,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "model_type": "mixtral",
    "torch_dtype": "bfloat16"
  }
EOF
  # docker exec -w / ci_sglang mkdir -p /dummy-grok
  # mkdir -p dummy-grok && wget https://sharkpublic.blob.core.windows.net/sharkpublic/sglang/dummy_grok.json -O dummy-grok/config.json
  # docker cp ./dummy-grok ci_sglang:/

  docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache huggingface_hub[hf_xet]
  docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache pytest

  # Install cache-dit for qwen_image_t2i_cache_dit_enabled test (added in PR 16204)
  docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache cache-dit || echo "cache-dit installation failed"

  # Install accelerate for distributed training and inference support
  docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache accelerate || echo "accelerate installation failed"
fi

if [[ -n "${SKIP_AITER_BUILD}" ]]; then
  exit 0
fi

# Detect AITER version
#############################################
# Detect correct AITER_COMMIT for this runner
# + Check mismatch
# + Rebuild AITER if needed
#############################################

echo "[CI-AITER-CHECK] === AITER VERSION CHECK START ==="

DOCKERFILE="docker/rocm.Dockerfile"

# GPU_ARCH
GPU_ARCH="${GPU_ARCH:-mi30x}"
echo "[CI-AITER-CHECK] Runner GPU_ARCH=${GPU_ARCH}"

#############################################
# 1. Extract AITER_COMMIT from correct Dockerfile block
#############################################
if [[ "${GPU_ARCH}" == "mi35x" ]]; then
    echo "[CI-AITER-CHECK] Using gfx950 block from Dockerfile..."
    REPO_AITER_COMMIT=$(grep -F -A20 'FROM $BASE_IMAGE_950 AS gfx950' docker/rocm.Dockerfile \
                        | grep 'AITER_COMMIT_DEFAULT=' \
                        | head -n1 \
                        | sed 's/.*AITER_COMMIT_DEFAULT="\([^"]*\)".*/\1/')
else
    echo "[CI-AITER-CHECK] Using gfx942 block from Dockerfile..."
    REPO_AITER_COMMIT=$(grep -F -A20 'FROM $BASE_IMAGE_942 AS gfx942' docker/rocm.Dockerfile \
                        | grep 'AITER_COMMIT_DEFAULT=' \
                        | head -n1 \
                        | sed 's/.*AITER_COMMIT_DEFAULT="\([^"]*\)".*/\1/')
fi


if [[ -z "${REPO_AITER_COMMIT}" ]]; then
    echo "[CI-AITER-CHECK] ERROR: Failed to extract AITER_COMMIT from Dockerfile."
    exit 1
fi

echo "[CI-AITER-CHECK] Dockerfile expects AITER_COMMIT=${REPO_AITER_COMMIT}"

#############################################
# 2. Check container pre-installed AITER version
#############################################
IMAGE_AITER_VERSION=$(docker exec ci_sglang bash -c "pip show amd-aiter 2>/dev/null | grep '^Version:' | awk '{print \$2}'" || echo "none")
IMAGE_AITER_VERSION="v${IMAGE_AITER_VERSION}"
echo "[CI-AITER-CHECK] AITER version inside CI image: ${IMAGE_AITER_VERSION}"

#############################################
# 3. Decide rebuild
#############################################
NEED_REBUILD="false"

if [[ -n "${AITER_COMMIT_OVERRIDE:-}" ]]; then
    echo "[CI-AITER-CHECK] AITER_COMMIT_OVERRIDE=${AITER_COMMIT_OVERRIDE} → forcing rebuild"
    REPO_AITER_COMMIT="${AITER_COMMIT_OVERRIDE}"
    NEED_REBUILD="true"
elif [[ "${IMAGE_AITER_VERSION}" == "vnone" || "${IMAGE_AITER_VERSION}" == "v" ]]; then
    echo "[CI-AITER-CHECK] No AITER found in image → rebuild needed"
    NEED_REBUILD="true"
elif [[ "${IMAGE_AITER_VERSION}" == "${REPO_AITER_COMMIT}" ]]; then
    echo "[CI-AITER-CHECK] AITER version matches"
elif [[ "${IMAGE_AITER_VERSION}" =~ (dev|\+g[0-9a-f]+) ]]; then
    # Dev/patched version (contains 'dev' or git hash) → preserve it
    echo "[CI-AITER-CHECK] Dev/patched version detected: ${IMAGE_AITER_VERSION} → skipping rebuild"
else
    echo "[CI-AITER-CHECK] Version mismatch: image=${IMAGE_AITER_VERSION}, repo=${REPO_AITER_COMMIT}"
    NEED_REBUILD="true"
fi


#############################################
# 4. Rebuild AITER if needed
#############################################
if [[ "${NEED_REBUILD}" == "true" ]]; then
    echo "[CI-AITER-CHECK] === AITER REBUILD START ==="

    # uninstall existing aiter
    docker exec ci_sglang pip uninstall -y amd-aiter || true

    # delete old aiter directory
    docker exec ci_sglang rm -rf /sgl-workspace/aiter

    # clone a fresh copy to /sgl-workspace/aiter
    docker exec ci_sglang git clone https://github.com/ROCm/aiter.git /sgl-workspace/aiter

    # checkout correct version and install requirements
    docker exec ci_sglang bash -c "
        cd /sgl-workspace/aiter && \
        git fetch --all && \
        git checkout ${REPO_AITER_COMMIT} && \
        git submodule update --init --recursive && \
        pip install -r requirements.txt
    "

    if [[ "${GPU_ARCH}" == "mi35x" ]]; then
        GPU_ARCH_LIST="gfx950"
    else
        GPU_ARCH_LIST="gfx942"
    fi
    echo "[CI-AITER-CHECK] GPU_ARCH_LIST=${GPU_ARCH_LIST}"

    # Re-apply Dockerfile hotpatches for ROCm 7.2 (the fresh clone lost them, can be removed after triton fixed this problem)
    ROCM_VERSION=$(docker exec ci_sglang bash -c "cat /opt/rocm/.info/version 2>/dev/null || echo unknown")
    if [[ "${ROCM_VERSION}" == 7.2* ]]; then
        echo "[CI-AITER-CHECK] ROCm 7.2 detected (${ROCM_VERSION}), applying AITER hotpatches..."
        docker exec ci_sglang bash -c "
            cd /sgl-workspace/aiter && \
            TARGET_FILE='aiter/ops/triton/attention/pa_mqa_logits.py' && \
            if [ -f \"\${TARGET_FILE}\" ]; then \
                sed -i '459 s/if.*:/if False:/' \"\${TARGET_FILE}\" && \
                echo '[CI-AITER-CHECK] Hotpatch applied to pa_mqa_logits.py'; \
            else \
                echo '[CI-AITER-CHECK] pa_mqa_logits.py not found, skipping hotpatch'; \
            fi
        "
    else
        echo "[CI-AITER-CHECK] ROCm version=${ROCM_VERSION}, no hotpatch needed"
    fi

    # build AITER
    docker exec ci_sglang bash -c "
        cd /sgl-workspace/aiter && \
        GPU_ARCHS=${GPU_ARCH_LIST} python3 setup.py develop
    "

    echo "[CI-AITER-CHECK] === AITER REBUILD COMPLETE ==="
fi

echo "[CI-AITER-CHECK] === AITER VERSION CHECK END ==="


# # Clear pre-built AITER kernels from Docker image to avoid segfaults
# # The Docker image may contain pre-compiled kernels incompatible with the current environment
# echo "Clearing pre-built AITER kernels from Docker image..."
# docker exec ci_sglang find /sgl-workspace/aiter/aiter/jit -name "*.so" -delete 2>/dev/null || true
# docker exec ci_sglang ls -la /sgl-workspace/aiter/aiter/jit/ 2>/dev/null || echo "jit dir empty or not found"

# # Pre-build AITER kernels to avoid timeout during tests
# echo "Warming up AITER JIT kernels..."
# docker exec -e SGLANG_USE_AITER=1 ci_sglang python3 /sglang-checkout/scripts/ci/amd/amd_ci_warmup_aiter.py || echo "AITER warmup completed (some kernels may not be available)"
