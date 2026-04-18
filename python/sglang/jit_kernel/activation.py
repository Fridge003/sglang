from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _fast_math_flags() -> list[str]:
    # Match sgl-kernel's CMake policy for activation.cu:
    # - SM90 build uses -use_fast_math
    # - SM100+ (Blackwell) builds use precise math — expf must match
    #   sgl_kernel's reference bit-for-bit, otherwise Llama-2 LoRA
    #   greedy generation diverges on B200 CI.
    # - nvcc-only; hipcc (ROCm, clang-based) rejects the flag.
    if is_hip_runtime():
        return []
    if get_jit_cuda_arch().major >= 10:
        return []
    return ["--use_fast_math"]


@cache_once
def _jit_silu_and_mul_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "silu_and_mul",
        *args,
        cuda_files=["elementwise/activation.cuh"],
        cuda_wrappers=[("silu_and_mul", f"SiluAndMulKernel<{args}>::run")],
        extra_cuda_cflags=_fast_math_flags(),
    )


@cache_once
def _jit_gelu_and_mul_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "gelu_and_mul",
        *args,
        cuda_files=["elementwise/activation.cuh"],
        cuda_wrappers=[("gelu_and_mul", f"GeluAndMulKernel<{args}>::run")],
        extra_cuda_cflags=_fast_math_flags(),
    )


def _prepare_out(input: torch.Tensor, out: Optional[torch.Tensor]) -> torch.Tensor:
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    else:
        assert input.dim() == out.dim(), f"{input.dim()} != {out.dim()}"
        assert (
            input.shape[:-1] == out.shape[:-1]
        ), f"{input.shape[:-1]} != {out.shape[:-1]}"
        assert (
            input.shape[-1] == 2 * out.shape[-1]
        ), f"{input.shape[-1]} != {2 * out.shape[-1]}"
    return out


@register_custom_op(op_name="jit_silu_and_mul", mutates_args=["out"])
def _silu_and_mul_custom_op(input: torch.Tensor, out: torch.Tensor) -> None:
    module = _jit_silu_and_mul_module(input.dtype)
    module.silu_and_mul(out.view(-1, out.shape[-1]), input.view(-1, input.shape[-1]))


@register_custom_op(op_name="jit_gelu_and_mul", mutates_args=["out"])
def _gelu_and_mul_custom_op(input: torch.Tensor, out: torch.Tensor) -> None:
    module = _jit_gelu_and_mul_module(input.dtype)
    module.gelu_and_mul(out.view(-1, out.shape[-1]), input.view(-1, input.shape[-1]))


def silu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    out = _prepare_out(input, out)
    _silu_and_mul_custom_op(input, out)
    return out


def gelu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    out = _prepare_out(input, out)
    _gelu_and_mul_custom_op(input, out)
    return out
