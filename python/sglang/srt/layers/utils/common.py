import logging
import re
from typing import Sequence

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# WeightTensor: unified wrapper for torch.Tensor and PySafeSlice
# ---------------------------------------------------------------------------


class WeightTensor:
    """Unified wrapper for :class:`torch.Tensor` and ``PySafeSlice``.

    ``PySafeSlice`` objects are returned by
    ``safetensors.safe_open.get_slice`` and support lazy, partial reads from
    disk.  This class provides a single API so that downstream weight-loader
    code never needs ``isinstance`` checks.

    Key methods
    -----------
    * ``shape`` / ``dtype`` – metadata without materializing.
    * ``get_tensor()``          – full materialization → ``torch.Tensor``.
    * ``get_narrowed_tensor(dim, start, size)`` – read only the requested
      shard from disk (for ``PySafeSlice``) or call ``.narrow()`` (for
      ``torch.Tensor``).  Always returns a ``torch.Tensor``.
    """

    __slots__ = ("_data", "_is_lazy")

    def __init__(self, data) -> None:
        self._data = data
        self._is_lazy = not isinstance(data, torch.Tensor)

    # -- metadata (no I/O) -------------------------------------------------

    @property
    def is_lazy(self) -> bool:
        """``True`` if the underlying data has not yet been materialized."""
        return self._is_lazy

    @property
    def shape(self) -> Sequence[int]:
        """Return shape without materializing."""
        if self._is_lazy:
            return self._data.get_shape()
        return self._data.shape

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype without materializing."""
        return self._data.dtype

    # -- materialization ----------------------------------------------------

    def get_tensor(self) -> torch.Tensor:
        """Return the full weight as a :class:`torch.Tensor`.

        No-op if already materialized.
        """
        if self._is_lazy:
            return self._data[:]
        return self._data

    def get_narrowed_tensor(
        self, dim: int, start: int, size: int
    ) -> torch.Tensor:
        """Narrow along *dim* and return a :class:`torch.Tensor`.

        For ``PySafeSlice`` the I/O reads only the requested byte range.
        For ``torch.Tensor`` this calls ``.narrow()`` directly.
        """
        if self._is_lazy:
            ndim = len(self._data.get_shape())
            idx = tuple(
                slice(start, start + size) if i == dim else slice(None)
                for i in range(ndim)
            )
            return self._data[idx]
        return self._data.narrow(dim, start, size)

    # -- fallback for non-migrated code ------------------------------------

    def __getattr__(self, name):
        """Auto-materialize and delegate for tensor attributes not on WeightTensor.

        This ensures backward compatibility with code that calls tensor methods
        (e.g. ``.view()``, ``.t()``, ``.to()``) directly on a weight without
        first calling ``.get_tensor()``.
        """
        return getattr(self.get_tensor(), name)

    def __getitem__(self, idx):
        """Support indexing — materializes lazy weights."""
        return self.get_tensor()[idx]


def pad_or_narrow_weight(
    loaded_weight: torch.Tensor, input_dim: int, start_idx: int, shard_size: int
) -> torch.Tensor:
    # Padding with zeros for special case such as qwen2_5_VL's mlp which is not 8-aligned
    valid_size = max(loaded_weight.shape[input_dim] - start_idx, 0)

    if valid_size > 0:
        loaded_slice = loaded_weight.narrow(input_dim, start_idx, valid_size)
        pad_shape = list(loaded_weight.shape)
        pad_shape[input_dim] = shard_size - valid_size
        pad = torch.zeros(
            pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
        )
        return torch.cat([loaded_slice, pad], dim=input_dim)

    # All padding
    pad_shape = list(loaded_weight.shape)
    pad_shape[input_dim] = shard_size
    return torch.zeros(
        pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
    )


def copy_or_rebind_param(
    module: torch.nn.Module, name: str, new_value: torch.Tensor
) -> None:
    """Keep parameter identities stable for CUDA graph reuse and hot reload."""
    new_value = new_value.detach()
    param = getattr(module, name, None)
    if isinstance(param, Parameter):
        if param.data.shape == new_value.shape and param.data.dtype == new_value.dtype:
            param.data.copy_(new_value)
        else:
            param.data = new_value
        param.requires_grad_(False)
    else:
        setattr(module, name, Parameter(new_value, requires_grad=False))


class PPMissingLayer(torch.nn.Identity):
    # Adapted from
    # https://github.com/vllm-project/vllm/blob/18ed3132d2bfe1df9a74729457b69243955221e8/vllm/model_executor/models/utils.py#L468C1-L486C1
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.return_tuple = kwargs.get("return_tuple", False)

    def forward(self, *args, **kwargs):
        """
        Return the first arg from args or the first value from kwargs.

        Wraps the input in a tuple if `self.return_tuple` is True.
        """
        input = args[0] if args else next(iter(kwargs.values()))
        return (input,) if self.return_tuple else input
