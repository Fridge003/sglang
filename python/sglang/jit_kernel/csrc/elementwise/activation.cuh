// Adapted from
// https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/include/flashinfer/activation.cuh

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace {

SGL_DEVICE float silu(const float& x) {
  return x / (1.0f + expf(-x));
}

SGL_DEVICE float gelu(const float& x) {
  constexpr float kAlpha = M_SQRT1_2;
  return x * 0.5f * (1.0f + erff(x * kAlpha));
}

template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input, const int d) {
  using namespace device;
  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = AlignedVector<T, vec_size>;

  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;

  const T* token_in = input + token_idx * 2 * d;
  T* token_out = out + token_idx * d;

#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    vec_t x_vec, y_vec, out_vec;
    x_vec.load(token_in, idx);
    y_vec.load(token_in + d, idx);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      const float xf = cast<float>(x_vec[i]);
      const float yf = cast<float>(y_vec[i]);
      out_vec[i] = cast<T>(Activation(xf) * yf);
    }
    out_vec.store(token_out, idx);
  }

  const int64_t remaining_offset = d - d % (stride * vec_size);
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    const float xf = cast<float>(token_in[remaining_offset + idx]);
    const float yf = cast<float>(token_in[d + remaining_offset + idx]);
    token_out[remaining_offset + idx] = cast<T>(Activation(xf) * yf);
  }
}

template <typename T, float (*Activation)(const float&)>
struct ActAndMulKernel {
  static void run(tvm::ffi::TensorView out, tvm::ffi::TensorView input) {
    using namespace host;

    auto num_tokens = SymbolicSize{"num_tokens"};
    auto half_d = SymbolicSize{"half_d"};
    auto full_d = SymbolicSize{"full_d"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({num_tokens, full_d})  //
        .with_dtype<T>()
        .with_device(device)
        .verify(input);
    TensorMatcher({num_tokens, half_d})  //
        .with_dtype<T>()
        .with_device(device)
        .verify(out);

    const int64_t n = static_cast<int64_t>(num_tokens.unwrap());
    const int d = static_cast<int>(half_d.unwrap());
    RuntimeCheck(
        static_cast<int64_t>(full_d.unwrap()) == 2LL * d,
        "act_and_mul: input last dim ",
        full_d.unwrap(),
        " must equal 2 * out last dim ",
        d);

    constexpr uint32_t vec_size = 16 / sizeof(T);
    const uint32_t block = static_cast<uint32_t>(std::min<int>(d / vec_size, 1024));
    dim3 grid(static_cast<uint32_t>(n));
    dim3 block_dim(block);

    LaunchKernel(grid, block_dim, device.unwrap())(
        act_and_mul_kernel<T, Activation>, static_cast<T*>(out.data_ptr()), static_cast<const T*>(input.data_ptr()), d);
  }
};

template <typename T>
struct SiluAndMulKernel : ActAndMulKernel<T, silu> {};

template <typename T>
struct GeluAndMulKernel : ActAndMulKernel<T, gelu> {};

}  // namespace
