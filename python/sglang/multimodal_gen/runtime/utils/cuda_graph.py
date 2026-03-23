from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Hashable

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors
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


def clone_tensor_preserve_layout(tensor: torch.Tensor) -> torch.Tensor:
    cloned = torch.empty_strided(
        size=tensor.shape,
        stride=tensor.stride(),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    cloned.copy_(tensor)
    return cloned


def ensure_tensor_signature_match(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    label: str,
    index: int,
) -> None:
    if (
        reference.shape != candidate.shape
        or reference.dtype != candidate.dtype
        or reference.device != candidate.device
        or reference.stride() != candidate.stride()
    ):
        raise ValueError(
            f"CUDA graph tensor signature mismatch for {label}[{index}]: "
            f"expected shape={tuple(reference.shape)} dtype={reference.dtype} "
            f"device={reference.device} stride={reference.stride()}, got "
            f"shape={tuple(candidate.shape)} dtype={candidate.dtype} "
            f"device={candidate.device} stride={candidate.stride()}"
        )


class SharedStaticInputPool:
    """Reusable static input buffers shared across captured graphs."""

    def __init__(self):
        self._buffers: dict[Hashable, tuple[torch.Tensor, ...]] = {}

    def clear(self) -> None:
        self._buffers.clear()

    def get_or_create(
        self, key: Hashable, example_inputs: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        buffers = self._buffers.get(key)
        if buffers is None:
            buffers = tuple(
                clone_tensor_preserve_layout(tensor) for tensor in example_inputs
            )
            self._buffers[key] = buffers
            return buffers

        if len(buffers) != len(example_inputs):
            raise ValueError(
                f"CUDA graph shared input buffer mismatch for key={key}: "
                f"expected {len(buffers)} tensors, got {len(example_inputs)}"
            )

        for index, (buffer, tensor) in enumerate(zip(buffers, example_inputs)):
            ensure_tensor_signature_match(
                buffer, tensor, label=f"shared_input[{key!r}]", index=index
            )
        return buffers


@dataclass
class CudaGraphCaptureEntry:
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[torch.Tensor, ...]
    output: Any


class CudaGraphCallableCache:
    """Small LRU cache for inference CUDA graphs to keep graph memory bounded."""

    def __init__(
        self,
        max_entries: int = 32,
        label: str | None = None,
        log_capture_events: bool = False,
        pool_handle: Any | None = None,
        shared_input_pool: SharedStaticInputPool | None = None,
        warmup_iters: int = 3,
    ):
        self._max_entries = max_entries
        self._cache: OrderedDict[Hashable, CudaGraphCaptureEntry] = OrderedDict()
        self._pool_handle = pool_handle
        self._label = label or "cuda-graph-cache"
        self._log_capture_events = log_capture_events
        self._shared_input_pool = shared_input_pool
        self._warmup_iters = warmup_iters

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
        input_buffer_key: Hashable | None = None,
    ) -> Any:
        if not all(isinstance(arg, torch.Tensor) for arg in example_inputs):
            raise TypeError(
                f"CUDA graph cache only supports tensor inputs, got key={key}"
            )
        tensor_inputs = tuple(example_inputs)

        entry = self._cache.get(key)
        if entry is None:
            if len(self._cache) >= self._max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                if self._log_capture_events:
                    logger.info(
                        "cuda graph cache evict: cache=%s key=%s",
                        self._label,
                        evicted_key,
                    )
            if input_buffer_key is not None and self._shared_input_pool is not None:
                static_inputs = self._shared_input_pool.get_or_create(
                    input_buffer_key, tensor_inputs
                )
            else:
                static_inputs = tuple(
                    clone_tensor_preserve_layout(tensor) for tensor in tensor_inputs
                )

            for static_input, tensor_input in zip(static_inputs, tensor_inputs):
                static_input.copy_(tensor_input)

            if self._warmup_iters > 0:
                torch.cuda.synchronize()
                warmup_stream = torch.cuda.Stream()
                with torch.cuda.stream(warmup_stream):
                    for _ in range(self._warmup_iters):
                        warmup_output = fn(*static_inputs)
                        del warmup_output
                warmup_stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._get_pool_handle()):
                output = fn(*static_inputs)

            entry = CudaGraphCaptureEntry(
                graph=graph,
                static_inputs=static_inputs,
                output=weak_ref_tensors(output),
            )
            self._cache[key] = entry
            if self._log_capture_events:
                logger.info(
                    "cuda graph capture miss: cache=%s key=%s entries=%d",
                    self._label,
                    key,
                    len(self._cache),
                )
            return output

        else:
            self._cache.move_to_end(key)

        inputs = tuple(example_inputs if call_inputs is None else call_inputs)
        if not all(isinstance(arg, torch.Tensor) for arg in inputs):
            raise TypeError(
                f"CUDA graph cache only supports tensor replay inputs, got key={key}"
            )

        if len(inputs) != len(entry.static_inputs):
            raise ValueError(
                f"CUDA graph replay input mismatch for key={key}: "
                f"expected {len(entry.static_inputs)} tensors, got {len(inputs)}"
            )

        for index, (static_input, tensor_input) in enumerate(
            zip(entry.static_inputs, inputs)
        ):
            ensure_tensor_signature_match(
                static_input, tensor_input, label=f"replay_input[{key!r}]", index=index
            )
            if static_input.data_ptr() != tensor_input.data_ptr():
                static_input.copy_(tensor_input)

        entry.graph.replay()
        return entry.output
