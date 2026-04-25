"""Unified component residency and offload coordination.

This module centralizes the runtime semantics around component residency:
whole-module CPU/GPU movement, layerwise windows, phase-specific managers, and
async readiness events.  Components describe what they can do; stages only
declare which components they need.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.layerwise_offload import (
    LayerwiseOffloadManager,
    OffloadableDiTMixin,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage

logger = init_logger(__name__)


class ComponentResidencyPolicy(str, Enum):
    LEGACY = "legacy"
    ADAPTIVE = "adaptive"
    RESIDENT_BIASED = "resident-biased"
    MEMORY_BIASED = "memory-biased"


class ComponentState(str, Enum):
    CPU_RESIDENT = "cpu_resident"
    CPU_SNAPSHOT = "cpu_snapshot"
    PREFETCHING = "prefetching"
    GPU_RESIDENT = "gpu_resident"
    LAYERWISE_RESIDENT_WINDOW = "layerwise_resident_window"
    RELEASING = "releasing"
    EXTERNAL = "external"


@dataclass(frozen=True)
class StageComponentDemand:
    required: tuple[str, ...] = ()
    preferred_after_request: tuple[str, ...] = ()
    peak_memory_class: str = "normal"


def normalize_component_residency_policy(policy: str | None) -> str:
    if policy is None:
        return ComponentResidencyPolicy.ADAPTIVE.value
    policy = policy.lower()
    valid = {item.value for item in ComponentResidencyPolicy}
    if policy not in valid:
        raise ValueError(
            f"Invalid component_residency_policy={policy!r}. Expected one of {sorted(valid)}."
        )
    return policy


def _first_parameter_or_buffer(module: torch.nn.Module) -> torch.Tensor | None:
    tensor = next(module.parameters(), None)
    if tensor is not None:
        return tensor
    return next(module.buffers(), None)


def _is_on_gpu(module: torch.nn.Module) -> bool:
    tensor = _first_parameter_or_buffer(module)
    return tensor is not None and tensor.device.type == current_platform.device_type


def _is_fsdp_managed(module: torch.nn.Module) -> bool:
    for submodule in module.modules():
        type_name = type(submodule).__name__
        if type_name == "FullyShardedDataParallel" or "FSDP" in type_name:
            return True
    return False


def _resolve_module_path(module: torch.nn.Module, path: str) -> torch.nn.Module | None:
    current: Any = module
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, torch.nn.Module) else None


def _candidate_layer_list_paths(module: torch.nn.Module) -> list[str]:
    preferred = (
        "layers",
        "blocks",
        "block",
        "transformer_blocks",
        "encoder.layers",
        "encoder.block",
        "model.layers",
    )
    paths: list[str] = []
    for path in preferred:
        candidate = _resolve_module_path(module, path)
        if isinstance(candidate, torch.nn.ModuleList) and len(candidate) > 1:
            paths.append(path)

    for name, submodule in module.named_modules():
        if not name or name in paths:
            continue
        if any(part.isdigit() for part in name.split(".")):
            continue
        if not isinstance(submodule, torch.nn.ModuleList) or len(submodule) < 4:
            continue
        if not name.endswith(("layers", "blocks", "block", "transformer_blocks")):
            continue
        paths.append(name)

    return paths


class ComponentResidencyAdapter:
    def __init__(
        self,
        name: str,
        module: torch.nn.Module | None,
        server_args: ServerArgs,
        *,
        memory_gb: float = 0.0,
        can_release: bool = False,
    ) -> None:
        self.name = name
        self.module = module
        self.server_args = server_args
        self.memory_gb = float(memory_gb or 0.0)
        self.can_release = can_release
        self.state = ComponentState.EXTERNAL if module is None else self._initial_state()

    def _initial_state(self) -> ComponentState:
        if self.module is not None and _is_on_gpu(self.module):
            return ComponentState.GPU_RESIDENT
        return ComponentState.CPU_RESIDENT

    @property
    def resident_memory_gb(self) -> float:
        return self.memory_gb

    def wait_ready(self) -> None:
        pass

    def prefetch(self, non_blocking: bool = True) -> None:
        pass

    def release(self) -> None:
        pass

    def prepare_for_next_request(self) -> None:
        self.prefetch(non_blocking=True)


class WholeModuleResidencyAdapter(ComponentResidencyAdapter):
    def __init__(
        self,
        name: str,
        module: torch.nn.Module,
        server_args: ServerArgs,
        *,
        memory_gb: float = 0.0,
        can_release: bool = False,
    ) -> None:
        super().__init__(
            name,
            module,
            server_args,
            memory_gb=memory_gb,
            can_release=can_release,
        )
        self._prefetch_stream = None
        self._ready_event = None
        self._fsdp_managed = _is_fsdp_managed(module)

    def _supports_async_transfer(self) -> bool:
        return current_platform.is_cuda() and torch.get_device_module().is_available()

    def _get_prefetch_stream(self):
        if not self._supports_async_transfer():
            return None
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.get_device_module().Stream(
                device=get_local_torch_device()
            )
        return self._prefetch_stream

    def wait_ready(self) -> None:
        if self._ready_event is not None and current_platform.is_cuda():
            torch.get_device_module().current_stream().wait_event(self._ready_event)
        self._ready_event = None

    def prefetch(self, non_blocking: bool = True) -> None:
        if self.module is None or self._fsdp_managed:
            return
        if _is_on_gpu(self.module):
            self.state = ComponentState.GPU_RESIDENT
            return

        stream = self._get_prefetch_stream()
        target_device = get_local_torch_device()
        if stream is None or not non_blocking:
            self.module.to(target_device, non_blocking=non_blocking)
            self.state = ComponentState.GPU_RESIDENT
            self._ready_event = None
            return

        with torch.get_device_module().stream(stream):
            self.module.to(target_device, non_blocking=True)
            event = torch.get_device_module().Event()
            event.record(stream)
        self._ready_event = event
        self.state = ComponentState.PREFETCHING

    def release(self) -> None:
        if (
            self.module is None
            or self._fsdp_managed
            or not self.can_release
            or not _is_on_gpu(self.module)
        ):
            return
        self.wait_ready()
        self.state = ComponentState.RELEASING
        self.module.to("cpu", non_blocking=True)
        self.state = ComponentState.CPU_RESIDENT


class LayerwiseResidencyAdapter(ComponentResidencyAdapter):
    def __init__(
        self,
        name: str,
        module: torch.nn.Module,
        server_args: ServerArgs,
        managers: list[LayerwiseOffloadManager],
        *,
        memory_gb: float = 0.0,
    ) -> None:
        super().__init__(
            name,
            module,
            server_args,
            memory_gb=memory_gb,
            can_release=True,
        )
        self.managers = managers
        self.state = ComponentState.LAYERWISE_RESIDENT_WINDOW

    @property
    def resident_memory_gb(self) -> float:
        if not self.managers:
            return 0.0
        ratio = 0.0
        for manager in self.managers:
            ratio += manager.prefetch_size / max(manager.num_layers, 1)
        ratio = min(1.0, ratio)
        return self.memory_gb * ratio

    @classmethod
    def from_existing(
        cls,
        name: str,
        module: torch.nn.Module,
        server_args: ServerArgs,
        *,
        memory_gb: float = 0.0,
    ) -> "LayerwiseResidencyAdapter | None":
        if not isinstance(module, OffloadableDiTMixin):
            return None
        managers = [
            manager
            for manager in getattr(module, "layerwise_offload_managers", [])
            if getattr(manager, "enabled", False)
        ]
        if not managers:
            return None
        return cls(name, module, server_args, managers, memory_gb=memory_gb)

    @classmethod
    def create_generic(
        cls,
        name: str,
        module: torch.nn.Module,
        server_args: ServerArgs,
        *,
        memory_gb: float = 0.0,
    ) -> "LayerwiseResidencyAdapter | None":
        if (
            not current_platform.is_cuda()
            or _is_fsdp_managed(module)
            or not _is_on_gpu(module)
        ):
            return None

        layer_paths = _candidate_layer_list_paths(module)
        if not layer_paths:
            return None

        managers = [
            LayerwiseOffloadManager(
                model=module,
                layers_attr_str=path,
                num_layers=len(_resolve_module_path(module, path)),  # type: ignore[arg-type]
                enabled=True,
                pin_cpu_memory=server_args.pin_cpu_memory,
                prefetch_size=1,
            )
            for path in layer_paths[:1]
        ]
        logger.info(
            "Enabled generic layerwise residency for %s on layer path(s): %s",
            name,
            layer_paths[:1],
        )
        return cls(name, module, server_args, managers, memory_gb=memory_gb)

    def wait_ready(self) -> None:
        for manager in self.managers:
            manager.prepare_for_next_req(non_blocking=False)
        self.state = ComponentState.LAYERWISE_RESIDENT_WINDOW

    def prefetch(self, non_blocking: bool = True) -> None:
        for manager in self.managers:
            manager.prepare_for_next_req(non_blocking=non_blocking)
        self.state = ComponentState.LAYERWISE_RESIDENT_WINDOW

    def release(self) -> None:
        for manager in self.managers:
            manager.release_all()
        self.state = ComponentState.CPU_RESIDENT

    def prepare_for_next_request(self) -> None:
        self.prefetch(non_blocking=True)


class ExternalPhaseResidencyAdapter(ComponentResidencyAdapter):
    """Adapter for existing phase managers such as LTX2 two-stage device mode."""

    def __init__(self, name: str, manager: Any, server_args: ServerArgs) -> None:
        super().__init__(name, None, server_args)
        self.manager = manager
        self.state = ComponentState.EXTERNAL

    def switch_phase(self, phase: str) -> bool:
        switch_phase = getattr(self.manager, "switch_phase", None)
        if callable(switch_phase):
            return bool(switch_phase(phase))
        return False

    def ensure_phase_ready(self, phase: str | None) -> None:
        ensure_phase_ready = getattr(self.manager, "ensure_phase_ready", None)
        if callable(ensure_phase_ready):
            ensure_phase_ready(phase)

    def prefetch_phase(self, phase: str) -> None:
        if phase == "stage2":
            prefetch_stage2 = getattr(self.manager, "prefetch_stage2_after_stage1", None)
            if callable(prefetch_stage2):
                prefetch_stage2()

    def release(self) -> None:
        release = getattr(self.manager, "release_premerged_transformers", None)
        if callable(release):
            release()

    def prepare_for_next_request(self) -> None:
        self.switch_phase("stage1")


class ComponentResidencyManager:
    def __init__(
        self,
        pipeline: "ComposedPipelineBase",
        server_args: ServerArgs,
    ) -> None:
        self.pipeline = pipeline
        self.server_args = server_args
        self.policy = normalize_component_residency_policy(
            server_args.component_residency_policy
        )
        self.margin_gb = float(server_args.component_residency_margin_gb)
        self.adapters: dict[str, ComponentResidencyAdapter] = {}
        self._initialized = False

    @property
    def enabled(self) -> bool:
        return self.policy != ComponentResidencyPolicy.LEGACY.value

    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self.enabled:
            return

        for name, module in self.pipeline.modules.items():
            if not isinstance(module, torch.nn.Module):
                continue
            adapter = self._build_adapter(name, module)
            if adapter is not None:
                self.adapters[name] = adapter

        device_manager = getattr(self.pipeline, "_device_manager", None)
        if device_manager is not None:
            self.adapters["ltx2_two_stage"] = ExternalPhaseResidencyAdapter(
                "ltx2_two_stage", device_manager, self.server_args
            )

        if self.adapters:
            logger.info(
                "Component residency manager initialized with policy=%s, components=%s",
                self.policy,
                sorted(self.adapters),
            )

    def _build_adapter(
        self, name: str, module: torch.nn.Module
    ) -> ComponentResidencyAdapter | None:
        memory_gb = self.pipeline.memory_usages.get(name, 0.0) or 0.0

        existing_layerwise = LayerwiseResidencyAdapter.from_existing(
            name, module, self.server_args, memory_gb=memory_gb
        )
        if existing_layerwise is not None:
            return existing_layerwise

        if self._should_try_generic_layerwise(name):
            generic_layerwise = LayerwiseResidencyAdapter.create_generic(
                name, module, self.server_args, memory_gb=memory_gb
            )
            if generic_layerwise is not None:
                return generic_layerwise

        return WholeModuleResidencyAdapter(
            name,
            module,
            self.server_args,
            memory_gb=memory_gb,
            can_release=self._offload_configured(name),
        )

    def _should_try_generic_layerwise(self, name: str) -> bool:
        if self.server_args.use_fsdp_inference:
            return False
        if name.startswith("text_encoder"):
            return bool(self.server_args.text_encoder_cpu_offload)
        if name.startswith("image_encoder"):
            return bool(self.server_args.image_encoder_cpu_offload)
        return False

    def _offload_configured(self, name: str) -> bool:
        if name.startswith("transformer") or name in {
            "video_dit",
            "video_dit_2",
            "audio_dit",
        }:
            return bool(
                self.server_args.dit_cpu_offload
                or self.server_args.dit_layerwise_offload
            )
        if name.startswith("text_encoder"):
            return bool(self.server_args.text_encoder_cpu_offload)
        if name.startswith("image_encoder"):
            return bool(self.server_args.image_encoder_cpu_offload)
        if name in {"vae", "video_vae", "audio_vae", "vocoder", "spatial_upsampler"}:
            return bool(self.server_args.vae_cpu_offload)
        return False

    def before_stage(
        self,
        stage_index: int,
        stage: "PipelineStage",
        stages: list["PipelineStage"],
        batch: "Req",
    ) -> None:
        del stage_index, stages, batch
        if not self.enabled:
            return

        demand = stage.component_demand()
        if demand.peak_memory_class == "high":
            self._release_for_peak(set(demand.required))

        for component in demand.required:
            adapter = self.adapters.get(component)
            if adapter is None:
                continue
            adapter.prefetch(non_blocking=True)
            adapter.wait_ready()

    def after_stage(
        self,
        stage_index: int,
        stage: "PipelineStage",
        stages: list["PipelineStage"],
        batch: "Req",
    ) -> None:
        del stage, batch
        if not self.enabled:
            return
        next_required = self._next_required_components(stage_index, stages)
        for component in next_required:
            self._prefetch_if_budget_allows(component)

    def after_request(self, batch: "Req") -> None:
        if not self.enabled:
            return
        preferred = self._next_request_preferred_components()
        if not preferred:
            return
        for component in preferred:
            self._prefetch_if_budget_allows(component)
        logger.debug(
            "Component residency after request (warmup=%s): %s",
            getattr(batch, "is_warmup", False),
            {
                name: adapter.state.value
                for name, adapter in self.adapters.items()
                if name in preferred
            },
        )

    def _next_required_components(
        self, stage_index: int, stages: list["PipelineStage"]
    ) -> tuple[str, ...]:
        for next_stage in stages[stage_index + 1 :]:
            demand = next_stage.component_demand()
            if demand.required:
                return demand.required
        return ()

    def _next_request_preferred_components(self) -> tuple[str, ...]:
        preferred: list[str] = []
        for stage in self.pipeline.stages:
            preferred.extend(stage.component_demand().preferred_after_request)

        for name, adapter in self.adapters.items():
            if isinstance(adapter, LayerwiseResidencyAdapter) and name not in preferred:
                preferred.append(name)
        return tuple(dict.fromkeys(preferred))

    def _available_budget_gb(self) -> float:
        if not current_platform.is_cuda():
            return 0.0
        try:
            return current_platform.get_available_gpu_memory(empty_cache=False)
        except Exception:
            return 0.0

    def _prefetch_if_budget_allows(self, component: str) -> None:
        adapter = self.adapters.get(component)
        if adapter is None:
            return
        if self.policy == ComponentResidencyPolicy.MEMORY_BIASED.value:
            if not isinstance(adapter, LayerwiseResidencyAdapter):
                return

        if self.policy != ComponentResidencyPolicy.RESIDENT_BIASED.value:
            budget_gb = self._available_budget_gb()
            if adapter.resident_memory_gb + self.margin_gb > budget_gb:
                return

        adapter.prepare_for_next_request()

    def _release_for_peak(self, required: set[str]) -> None:
        if self.policy == ComponentResidencyPolicy.RESIDENT_BIASED.value:
            return
        for name, adapter in self.adapters.items():
            if name in required:
                continue
            if isinstance(adapter, LayerwiseResidencyAdapter):
                adapter.release()

    def switch_external_phase(self, component: str, phase: str) -> bool:
        adapter = self.adapters.get(component)
        if not isinstance(adapter, ExternalPhaseResidencyAdapter):
            return False
        return adapter.switch_phase(phase)

    def ensure_external_phase_ready(self, component: str, phase: str | None) -> bool:
        adapter = self.adapters.get(component)
        if not isinstance(adapter, ExternalPhaseResidencyAdapter):
            return False
        adapter.ensure_phase_ready(phase)
        return True

    def prefetch_external_phase(self, component: str, phase: str) -> bool:
        adapter = self.adapters.get(component)
        if not isinstance(adapter, ExternalPhaseResidencyAdapter):
            return False
        adapter.prefetch_phase(phase)
        return True

    def release_external_component(self, component: str) -> bool:
        adapter = self.adapters.get(component)
        if not isinstance(adapter, ExternalPhaseResidencyAdapter):
            return False
        adapter.release()
        return True
