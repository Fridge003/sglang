/// \file warp.cuh
/// \brief Warp-level reduction primitives using `__shfl_xor_sync`.

#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>

namespace device::warp {

/// \brief Full 32-thread active mask.
static constexpr uint32_t kFullMask = 0xffffffffu;

/**
 * \brief Warp-level sum reduction.
 *
 * Computes the sum of `value` across all active lanes specified by
 * `active_mask` using butterfly (XOR) shuffles. The result is
 * broadcast to all participating lanes.
 *
 * \tparam kThreads Group size for the reduction (defaults to a full warp).
 * \tparam T Numeric type (e.g. float).
 * \param value Per-lane input value.
 * \param active_mask Bitmask of participating lanes (default: all 32).
 * \return The sum across all active lanes.
 */
template <uint32_t kThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_sum(T value, uint32_t active_mask = kFullMask) {
#pragma unroll
  for (auto mask = kThreads >> 1; mask > 0; mask >>= 1)
    value = value + __shfl_xor_sync(active_mask, value, mask, 32);
  return value;
}

/**
 * \brief Warp-level max reduction.
 *
 * Computes the maximum of `value` across all active lanes using
 * butterfly shuffles. The result is broadcast to all participating
 * lanes.
 *
 * \tparam kThreads Group size for the reduction (defaults to a full warp).
 * \tparam T Numeric type (must be supported by `math::max`).
 * \param value Per-lane input value.
 * \param active_mask Bitmask of participating lanes (default: all 32).
 * \return The maximum across all active lanes.
 */
template <uint32_t kThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_max(T value, uint32_t active_mask = kFullMask) {
#pragma unroll
  for (auto mask = kThreads >> 1; mask > 0; mask >>= 1)
    value = math::max(value, __shfl_xor_sync(active_mask, value, mask, 32));
  return value;
}

}  // namespace device::warp
