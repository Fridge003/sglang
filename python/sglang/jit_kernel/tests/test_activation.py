import itertools
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

DIM_LIST = get_ci_test_range(
    [128, 256, 512, 2048, 4096, 11008, 16384],
    [128, 2048, 16384],
)
BATCH_LIST = get_ci_test_range([1, 2, 4, 8, 16], [1, 8])
SEQ_LIST = get_ci_test_range([1, 2, 4, 8, 16, 32, 64, 128, 512], [1, 128])
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize(
    "dtype,dim,batch_size,seq_len",
    list(itertools.product(DTYPES, DIM_LIST, BATCH_LIST, SEQ_LIST)),
)
def test_silu_and_mul(dtype, dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device="cuda")
    y_ref = F.silu(x[..., :dim]) * x[..., dim:]
    y = silu_and_mul(x)
    tol = 1e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(y, y_ref, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "dtype,dim,batch_size,seq_len",
    list(itertools.product(DTYPES, DIM_LIST, BATCH_LIST, SEQ_LIST)),
)
def test_gelu_and_mul(dtype, dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device="cuda")
    y_ref = F.gelu(x[..., :dim], approximate="none") * x[..., dim:]
    y = gelu_and_mul(x)
    tol = 1e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(y, y_ref, rtol=tol, atol=tol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
