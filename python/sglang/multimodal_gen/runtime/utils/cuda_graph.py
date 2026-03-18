from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Hashable

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils.common import next_power_of_2

logger = init_logger(__name__)


def pad_tensor_along_dim(
    tensor: torch.Tensor, dim: int, target_length: int
) -> torch.Tensor:
    padding_length = target_length - tensor.shape[dim]
    if padding_length == 0:
        return tensor
    if padding_length < 0:
        raise ValueError(
            f"Target length={target_length} is smaller than current length={tensor.shape[dim]}"
        )

    padding_shape = list(tensor.shape)
    padding_shape[dim] = padding_length
    padding = tensor.new_zeros(tuple(padding_shape))
    return torch.cat([tensor, padding], dim=dim)


def pad_tensor_to_next_power_of_2(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    seq_length = tensor.shape[dim]
    target_length = next_power_of_2(seq_length)
    return pad_tensor_along_dim(tensor, dim=dim, target_length=target_length)


def replace_shape_dim(shape: Sequence[int], dim: int, value: int) -> tuple[int, ...]:
    updated_shape = list(shape)
    updated_shape[dim] = value
    return tuple(updated_shape)


def shape_with_next_power_of_2(shape: Sequence[int], dim: int) -> tuple[int, ...]:
    return replace_shape_dim(shape, dim=dim, value=next_power_of_2(shape[dim]))


class CudaGraphCallableCache:
    """Small LRU cache for graphed callables to keep graph memory bounded."""

    def __init__(
        self,
        max_entries: int = 32,
        label: str | None = None,
        log_capture_events: bool = False,
    ):
        self._max_entries = max_entries
        self._cache: OrderedDict[Hashable, Callable[..., Any]] = OrderedDict()
        self._pool_handle = None
        self._label = label or "cuda-graph-cache"
        self._log_capture_events = log_capture_events

    def clear(self) -> None:
        self._cache.clear()

    def _get_pool_handle(self):
        if self._pool_handle is None:
            self._pool_handle = torch.cuda.graphs.graph_pool_handle()
        return self._pool_handle

    def capture_or_replay(
        self,
        key: Hashable,
        fn: Callable[..., Any],
        example_inputs: Sequence[Any],
        call_inputs: Sequence[Any] | None = None,
    ) -> Any:
        module = self._cache.get(key)
        if module is None:
            if len(self._cache) >= self._max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                if self._log_capture_events:
                    logger.info(
                        "cuda graph cache evict: cache=%s key=%s",
                        self._label,
                        evicted_key,
                    )
            module = torch.cuda.make_graphed_callables(
                fn,
                tuple(example_inputs),
                pool=self._get_pool_handle(),
            )
            self._cache[key] = module
            if self._log_capture_events:
                logger.info(
                    "cuda graph capture miss: cache=%s key=%s entries=%d",
                    self._label,
                    key,
                    len(self._cache),
                )
        else:
            self._cache.move_to_end(key)

        inputs = tuple(example_inputs if call_inputs is None else call_inputs)
        return module(*inputs)
