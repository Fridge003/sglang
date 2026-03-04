#!/bin/bash
set -e

# CCCL symlink for CUDA 13 cutlass headers
ln -sf /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda

# Install cmake if missing
command -v cmake >/dev/null || pip install cmake

# Cherry-pick GB300 FlashMLA patch from upstream sglang
cd /sgl-workspace/sglang
if ! grep -q "SM103" sgl-kernel/cmake/flashmla.cmake; then
    git fetch https://github.com/sgl-project/sglang.git 8acd4d7d7e6f436601ef3ae51678f130fdc04d25
    git cherry-pick --no-commit FETCH_HEAD
    echo "Applied GB300 FlashMLA patch"
else
    echo "GB300 FlashMLA patch already applied"
fi

# Build flashmla_ops standalone
cd sgl-kernel
mkdir -p build_fmla
cd build_fmla
ln -sf ../csrc csrc
ln -sf ../include include

cat > CMakeLists.txt << 'CMAKE_EOF'
cmake_minimum_required(VERSION 3.21)
project(fmla LANGUAGES CXX CUDA)
find_package(Python COMPONENTS Interpreter Development.SABIModule REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(CUDA_VERSION ${CUDAToolkit_VERSION})
set(SKBUILD_SABI_VERSION 3.8)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../include ${CMAKE_CURRENT_LIST_DIR}/../csrc)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/flashmla.cmake)
CMAKE_EOF

rm -rf build
TORCH_CUDA_ARCH_LIST="9.0a;10.0a;10.3a" \
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -Wno-dev

cmake --build build -j$(nproc)

# Replace installed flashmla_ops (avoid importing sgl_kernel — no GPU during docker build)
DEST=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/sgl_kernel
cp build/flashmla_ops.abi3.so "$DEST/flashmla_ops.abi3.so"
echo "FlashMLA GB300 build complete: $DEST/flashmla_ops.abi3.so"
