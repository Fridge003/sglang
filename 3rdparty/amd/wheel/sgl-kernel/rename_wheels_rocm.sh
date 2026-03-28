#!/usr/bin/env bash
# Align ROCm wheel filenames (+rocmXXX) with internal METADATA Version and WHEEL tags
# after build (fixes pip "inconsistent version" when only the .whl name changed).
# Unpack → patch WHEEL/METADATA → wheel pack (RECORD regenerated; no hand-editing).
set -ex

WHEEL_DIR="dist"

detect_rocm_suffix() {
    local rocm_dir
    rocm_dir=$(realpath /opt/rocm 2>/dev/null || realpath /opt/rocm-* 2>/dev/null | head -1)
    if [[ -z "$rocm_dir" ]]; then
        echo ""
        return
    fi
    local ver_abrv
    ver_abrv=$(basename "$rocm_dir" | sed -e 's/rocm-//' -e 's/\./\//g' | tr -d '/')
    echo "+rocm${ver_abrv}"
}

ROCM_SUFFIX=$(detect_rocm_suffix)

patch_wheel_platform_tags() {
    local wheel_file="$1"
    sed -i \
        -e 's/-linux_x86_64$/-manylinux2014_x86_64/' \
        -e 's/-linux_aarch64$/-manylinux2014_aarch64/' \
        "$wheel_file"
}

wheel_files=("$WHEEL_DIR"/*.whl)
for wheel in "${wheel_files[@]}"; do
    [[ -f "$wheel" ]] || continue
    [[ "$wheel" == *"+rocm"* ]] && continue

    if [[ -z "$ROCM_SUFFIX" ]]; then
        continue
    fi

    TMPDIR=$(mktemp -d)
    trap 'rm -rf -- "$TMPDIR"' ERR

    python3 -m wheel unpack "$wheel" --dest "$TMPDIR"
    UNPACKED=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d | head -1)
    DIST_INFO=$(find "$UNPACKED" -maxdepth 1 -type d -name "*.dist-info" | head -1)
    WHEEL_META="${DIST_INFO}/WHEEL"
    METADATA_FILE="${DIST_INFO}/METADATA"

    patch_wheel_platform_tags "$WHEEL_META"

    ORIG_VERSION=$(grep '^Version:' "$METADATA_FILE" | head -1 | sed 's/^Version:[[:space:]]*//')
    if [[ "$ORIG_VERSION" == *"$ROCM_SUFFIX"* ]]; then
        echo "Skipping $wheel: version in METADATA is already suffixed."
        rm -rf "$TMPDIR"
        trap - ERR
        continue
    fi
    NEW_VERSION="${ORIG_VERSION}${ROCM_SUFFIX}"
    sed -i "s/^Version:.*/Version: ${NEW_VERSION}/" "$METADATA_FILE"

    OLD_BASE=$(basename "$DIST_INFO")
    NEW_BASE="${OLD_BASE/${ORIG_VERSION}/${NEW_VERSION}}"
    mv "$DIST_INFO" "${UNPACKED}/${NEW_BASE}"

    rm -f "$wheel"
    python3 -m wheel pack "$UNPACKED" --dest-dir "$WHEEL_DIR"
    rm -rf "$TMPDIR"
    trap - ERR
done
echo "Wheel renaming completed."
