#!/usr/bin/env bash
set -eo pipefail

usage() {
  echo "Usage: $0 --pat <GH_PAT> --tag <image_tag> --commits <sha1,sha2,...>"
  echo ""
  echo "  --pat      GitHub personal access token"
  echo "  --tag      Docker image tag (e.g. glm5-grace-blackwell)"
  echo "  --commits  Comma-separated commit SHAs to cherry-pick"
  echo "  --dry-run  Do everything except push the image"
  exit 1
}

PAT=""
IMAGE_TAG=""
COMMITS=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pat)      PAT="$2"; shift 2 ;;
    --tag)      IMAGE_TAG="$2"; shift 2 ;;
    --commits)  COMMITS="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=true; shift ;;
    *)          usage ;;
  esac
done

if [ -z "${PAT}" ] || [ -z "${IMAGE_TAG}" ] || [ -z "${COMMITS}" ]; then
  usage
fi

IMAGE="lmsysorg/sglang:${IMAGE_TAG}"
GIT=(git -C /sgl-workspace/sglang)
CONTAINER_ID=""

cleanup() {
  if [ -n "${CONTAINER_ID}" ]; then
    echo "Cleaning up container ${CONTAINER_ID:0:12}..."
    docker rm -f "${CONTAINER_ID}" > /dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "=== Pulling ${IMAGE} ==="
docker pull "${IMAGE}"

echo "=== Starting container ==="
CONTAINER_ID=$(docker run -d "${IMAGE}" sleep infinity)
echo "Container: ${CONTAINER_ID:0:12}"

# Extract base commit
BASE_SHA=$(docker exec "${CONTAINER_ID}" git -C /sgl-workspace/sglang rev-parse HEAD 2>/dev/null | tr -d '[:space:]') || true
if [ -z "${BASE_SHA}" ]; then
  echo "ERROR: Image has no .git directory"
  exit 1
fi
echo "Image built from commit: ${BASE_SHA}"

# Configure git inside container
docker exec "${CONTAINER_ID}" "${GIT[@]}" config user.email "ci@sglang.ai"
docker exec "${CONTAINER_ID}" "${GIT[@]}" config user.name "SGLang CI"

# Remove any baked-in auth headers from the image (e.g. GitHub Actions extraheader)
docker exec "${CONTAINER_ID}" "${GIT[@]}" config --unset-all http.https://github.com/.extraheader 2>/dev/null || true

# Set up authenticated remote URL
AUTHED_URL="https://x-access-token:${PAT}@github.com/sgl-project/sglang.git"
docker exec "${CONTAINER_ID}" "${GIT[@]}" remote set-url origin "${AUTHED_URL}" \
  || docker exec "${CONTAINER_ID}" "${GIT[@]}" remote add origin "${AUTHED_URL}"

# Verify the URL was set correctly (redact token in output)
ACTUAL_URL=$(docker exec "${CONTAINER_ID}" "${GIT[@]}" remote get-url origin)
if [[ "${ACTUAL_URL}" != *"x-access-token"* ]]; then
  echo "ERROR: Failed to set authenticated remote URL"
  echo "Got: ${ACTUAL_URL}"
  exit 1
fi
echo "Remote URL configured (token embedded)"

# Cherry-pick each commit
APPLIED=0
IFS=',' read -ra SHAS <<< "${COMMITS}"
for sha in "${SHAS[@]}"; do
  sha=$(echo "${sha}" | xargs)
  echo ""
  echo "=== Cherry-picking ${sha} ==="

  # Fetch this specific commit
  docker exec "${CONTAINER_ID}" "${GIT[@]}" fetch origin "${sha}"

  # Verify it exists
  if ! docker exec "${CONTAINER_ID}" "${GIT[@]}" cat-file -e "${sha}^{commit}" 2>/dev/null; then
    echo "ERROR: Commit ${sha} not found"
    exit 1
  fi

  # Reject merge commits
  PARENT_COUNT=$(docker exec "${CONTAINER_ID}" "${GIT[@]}" cat-file -p "${sha}" | grep -c '^parent' || true)
  if [ "${PARENT_COUNT}" -gt 1 ]; then
    echo "ERROR: ${sha} is a merge commit — use individual commit SHAs instead"
    exit 1
  fi

  # Skip if already applied
  if docker exec "${CONTAINER_ID}" "${GIT[@]}" merge-base --is-ancestor "${sha}" HEAD 2>/dev/null; then
    echo "  Already in image — skipping"
    continue
  fi

  docker exec "${CONTAINER_ID}" "${GIT[@]}" cherry-pick "${sha}" \
    --strategy=ort --strategy-option=theirs --no-edit
  APPLIED=$((APPLIED + 1))
  echo "  Applied successfully"
done

echo ""
if [ "${APPLIED}" -eq 0 ]; then
  echo "All commits already in image — nothing to do"
  exit 0
fi

echo "Applied ${APPLIED} commit(s)"

# Scrub credentials from remote URL
docker exec "${CONTAINER_ID}" "${GIT[@]}" remote set-url origin "https://github.com/sgl-project/sglang.git"

# Show final state
NEW_SHA=$(docker exec "${CONTAINER_ID}" "${GIT[@]}" rev-parse HEAD | tr -d '[:space:]')
echo "New HEAD: ${NEW_SHA}"

# Commit the container as the new image
echo ""
echo "=== Committing container as ${IMAGE} ==="
docker commit "${CONTAINER_ID}" "${IMAGE}"

if [ "${DRY_RUN}" = true ]; then
  echo "Dry run — skipping push"
else
  echo "=== Pushing ${IMAGE} ==="
  docker push "${IMAGE}"
  echo "Done!"
fi
