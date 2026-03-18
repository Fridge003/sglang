from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Hashable

import torch

from sglang.srt.utils.common import next_power_of_2


def pad_tensor_to_length(
    tensor: torch.Tensor, dim: int, padded_length: int
) -> torch.Tensor:
    padding_length = padded_length - tensor.shape[dim]
    if padding_length == 0:
        return tensor
    if padding_length < 0:
        raise ValueError(
            f"Target padded_length={padded_length} is smaller than current length={tensor.shape[dim]}"
        )

    padding_shape = list(tensor.shape)
    padding_shape[dim] = padding_length
    padding = tensor.new_zeros(tuple(padding_shape))
    return torch.cat([tensor, padding], dim=dim)


def pad_tensor_to_power_of_2(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    seq_length = tensor.shape[dim]
    padded_length = next_power_of_2(seq_length)
    return pad_tensor_to_length(tensor, dim=dim, padded_length=padded_length)


def shape_with_dim(shape: Sequence[int], dim: int, value: int) -> tuple[int, ...]:
    padded_shape = list(shape)
    padded_shape[dim] = value
    return tuple(padded_shape)


def power_of_2_shape(shape: Sequence[int], dim: int) -> tuple[int, ...]:
    return shape_with_dim(shape, dim=dim, value=next_power_of_2(shape[dim]))


class CudaGraphCallableCache:
    """Small LRU cache for graphed callables to keep graph memory bounded."""

    def __init__(self, max_entries: int = 32):
        self._max_entries = max_entries
        self._cache: OrderedDict[Hashable, Callable[..., Any]] = OrderedDict()
        self._pool_handle = None

    def clear(self) -> None:
        self._cache.clear()

    def _get_pool_handle(self):
        if self._pool_handle is None:
            self._pool_handle = torch.cuda.graphs.graph_pool_handle()
        return self._pool_handle

    def run(
        self,
        key: Hashable,
        fn: Callable[..., Any],
        example_inputs: Sequence[Any],
        call_inputs: Sequence[Any] | None = None,
    ) -> Any:
        module = self._cache.get(key)
        if module is None:
            if len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
            module = torch.cuda.make_graphed_callables(
                fn,
                tuple(example_inputs),
                pool=self._get_pool_handle(),
            )
            self._cache[key] = module
        else:
            self._cache.move_to_end(key)

        inputs = tuple(example_inputs if call_inputs is None else call_inputs)
        return module(*inputs)
