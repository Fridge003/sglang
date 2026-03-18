import math

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    LTX2AVDecodingStage,
    LTX2AVDenoisingStage,
    LTX2AVLatentPreparationStage,
    LTX2HalveResolutionStage,
    LTX2LoRASwitchStage,
    LTX2RefinementStage,
    LTX2TextConnectorStage,
    LTX2UpsampleStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.tensor_trace import get_trace_writer

logger = init_logger(__name__)

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: ServerArgs):
    vae_arch = getattr(
        getattr(server_args.pipeline_config, "vae_config", None), "arch_config", None
    )
    vae_scale_factor = int(
        getattr(vae_arch, "spatial_compression_ratio", None)
        or getattr(vae_arch, "vae_scale_factor", None)
        or getattr(server_args.pipeline_config, "vae_scale_factor", None)
        or 32
    )
    vae_temporal_compression = int(
        getattr(vae_arch, "temporal_compression_ratio", None)
        or getattr(server_args.pipeline_config, "vae_temporal_compression", None)
        or 8
    )
    latent_height = max(1, int(batch.height) // vae_scale_factor)
    latent_width = max(1, int(batch.width) // vae_scale_factor)
    latent_frames = max(1, (int(batch.num_frames) - 1) // vae_temporal_compression + 1)
    image_seq_len = latent_height * latent_width * latent_frames
    image_seq_len = max(1024, min(4096, image_seq_len))

    mu = calculate_shift(
        image_seq_len,
        base_seq_len=1024,
        max_seq_len=4096,
        base_shift=0.95,
        max_shift=2.05,
    )
    return "mu", mu


def build_ltx2_native_sigmas(
    batch: Req,
    server_args: ServerArgs,
    *,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.FloatTensor:
    # Copied and adapted from /root/LTX-2/packages/ltx-core/src/ltx_core/components/schedulers.py
    _ = server_args
    tokens = MAX_SHIFT_ANCHOR

    sigmas = torch.linspace(1.0, 0.0, int(batch.num_inference_steps) + 1)

    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    sigma_shift = tokens * mm + b
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
        0,
    )

    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    return sigmas.to(torch.float32)


class LTX2SigmaPreparationStage(PipelineStage):
    """Prepare native LTX-2 sigma schedule before timestep setup."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra["ltx2_trace_stage"] = "stage1"
        sigmas = build_ltx2_native_sigmas(batch, server_args)
        batch.sigmas = sigmas[:-1].tolist()
        writer = get_trace_writer(batch, server_args)
        writer.trace_tensor(
            event="stage1.schedule.sigmas",
            stage="stage1",
            tensor_name="sigmas",
            tensor=sigmas,
            metadata={"num_inference_steps": batch.num_inference_steps},
        )
        return batch


class LTX2RequestTraceStage(PipelineStage):
    """Emit request-level trace metadata for LTX-2 alignment."""

    def __init__(self, pipeline: ComposedPipelineBase):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        writer = get_trace_writer(batch, server_args)
        request_metadata = {
            "prompt": batch.prompt,
            "negative_prompt": batch.negative_prompt,
            "seed": getattr(batch, "seed", None),
            "height": batch.height,
            "width": batch.width,
            "num_frames": batch.num_frames,
            "fps": getattr(batch, "fps", None),
            "num_inference_steps": batch.num_inference_steps,
            "guidance_scale": batch.guidance_scale,
            "do_cfg": batch.do_classifier_free_guidance,
            "images": getattr(batch, "image_path", None)
            or getattr(batch, "images", None),
            "base_lora_path": server_args.lora_path,
            "base_lora_scale": server_args.lora_scale,
            "distilled_lora_path": server_args.component_paths.get("distilled_lora"),
            "distilled_lora_scale_configured": getattr(
                server_args, "distilled_lora_scale", 1.0
            ),
            "spatial_upsampler_path": server_args.component_paths.get(
                "spatial_upsampler"
            ),
            "pipeline_name": getattr(self.pipeline, "pipeline_name", None),
        }
        writer.trace_metadata(
            event="request.inputs",
            stage="stage1",
            metadata=request_metadata,
        )
        return batch


class LTX2TimestepTraceStage(PipelineStage):
    """Emit timestep trace metadata after scheduler setup."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        writer = get_trace_writer(batch, server_args)
        if isinstance(batch.timesteps, torch.Tensor):
            writer.trace_tensor(
                event="stage1.schedule.timesteps",
                stage="stage1",
                tensor_name="timesteps",
                tensor=batch.timesteps,
            )
        return batch


def _add_ltx2_front_stages(pipeline: ComposedPipelineBase):
    pipeline.add_stages(
        [
            InputValidationStage(),
            LTX2RequestTraceStage(pipeline=pipeline),
            TextEncodingStage(
                text_encoders=[pipeline.get_module("text_encoder")],
                tokenizers=[pipeline.get_module("tokenizer")],
            ),
            LTX2TextConnectorStage(connectors=pipeline.get_module("connectors")),
        ]
    )


def _add_ltx2_stage1_generation_stages(pipeline: ComposedPipelineBase):
    pipeline.add_stage(LTX2SigmaPreparationStage())
    pipeline.add_standard_timestep_preparation_stage()
    pipeline.add_stage(LTX2TimestepTraceStage())
    pipeline.add_stages(
        [
            LTX2AVLatentPreparationStage(
                scheduler=pipeline.get_module("scheduler"),
                transformer=pipeline.get_module("transformer"),
                audio_vae=pipeline.get_module("audio_vae"),
            ),
            LTX2AVDenoisingStage(
                transformer=pipeline.get_module("transformer"),
                scheduler=pipeline.get_module("scheduler"),
                vae=pipeline.get_module("vae"),
                audio_vae=pipeline.get_module("audio_vae"),
            ),
        ]
    )


def _add_ltx2_decoding_stage(pipeline: ComposedPipelineBase):
    pipeline.add_stage(
        LTX2AVDecodingStage(
            vae=pipeline.get_module("vae"),
            audio_vae=pipeline.get_module("audio_vae"),
            vocoder=pipeline.get_module("vocoder"),
            pipeline=pipeline,
        )
    )


class LTX2FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """Override ``_time_shift_exponential`` to use torch f32 instead of numpy f64."""

    def set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
    ):
        if sigmas is not None and timesteps is None:
            sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
            self.num_inference_steps = len(timesteps)
            self.timesteps = timesteps
            self.sigmas = sigmas
            self._step_index = None
            self._begin_index = None
            return

        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )

    def _time_shift_exponential(self, mu, sigma, t):
        if isinstance(t, np.ndarray):
            t_torch = torch.from_numpy(t).to(torch.float32)
            result = math.exp(mu) / (math.exp(mu) + (1 / t_torch - 1) ** sigma)
            return result.numpy()
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class _BaseLTX2Pipeline(LoRAPipeline):
    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        orig = self.get_module("scheduler")
        self.modules["scheduler"] = LTX2FlowMatchScheduler.from_config(orig.config)


class LTX2Pipeline(_BaseLTX2Pipeline):
    # Must match model_index.json `_class_name`.
    pipeline_name = "LTX2Pipeline"

    def create_pipeline_stages(self, server_args: ServerArgs):
        _add_ltx2_front_stages(self)
        _add_ltx2_stage1_generation_stages(self)
        _add_ltx2_decoding_stage(self)


class LTX2TwoStagePipeline(_BaseLTX2Pipeline):
    pipeline_name = "LTX2TwoStagePipeline"
    STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

    def initialize_pipeline(self, server_args: ServerArgs):
        super().initialize_pipeline(server_args)
        upsampler_path = server_args.component_paths.get("spatial_upsampler")
        if not upsampler_path:
            raise ValueError(
                "LTX2TwoStagePipeline requires --spatial-upsampler-path "
                "(component_paths['spatial_upsampler'])."
            )
        module, memory_usage = PipelineComponentLoader.load_component(
            component_name="spatial_upsampler",
            component_model_path=upsampler_path,
            transformers_or_diffusers="diffusers",
            server_args=server_args,
        )
        self.modules["spatial_upsampler"] = module
        self.memory_usages["spatial_upsampler"] = memory_usage

        distilled_lora_path = server_args.component_paths.get("distilled_lora")
        if not distilled_lora_path:
            raise ValueError(
                "LTX2TwoStagePipeline requires --distilled-lora-path "
                "(component_paths['distilled_lora'])."
            )
        self._distilled_lora_path = distilled_lora_path
        self._configured_distilled_lora_scale = float(
            getattr(server_args, "distilled_lora_scale", 1.0)
        )
        self._effective_distilled_lora_scale = 1.0
        self._stage1_lora_path = server_args.lora_path
        self._stage1_lora_scale = float(server_args.lora_scale)
        self._active_lora_phase = None

    def get_ltx2_trace_lora_state(self) -> dict[str, object]:
        return {
            "active_phase": self._active_lora_phase,
            "base_lora_path": self._stage1_lora_path,
            "base_lora_scale": self._stage1_lora_scale,
            "distilled_lora_path": self._distilled_lora_path,
            "distilled_lora_scale_configured": self._configured_distilled_lora_scale,
            "distilled_lora_scale_effective": self._effective_distilled_lora_scale,
            "adapter_config": self.cur_adapter_config.get("transformer"),
            "adapter_name": self.cur_adapter_name.get("transformer"),
            "adapter_path": self.cur_adapter_path.get("transformer"),
            "adapter_strength": self.cur_adapter_strength.get("transformer"),
        }

    def switch_lora_phase(self, phase: str) -> None:
        if phase == self._active_lora_phase:
            return

        if phase == "stage1":
            if self._stage1_lora_path:
                self.set_lora(
                    lora_nickname="ltx2_stage1_base",
                    lora_path=self._stage1_lora_path,
                    target="transformer",
                    strength=self._stage1_lora_scale,
                )
            else:
                self.unmerge_lora_weights(target="transformer")
        elif phase == "stage2":
            lora_nicknames = []
            lora_paths = []
            lora_strengths = []
            lora_targets = []
            if self._stage1_lora_path:
                lora_nicknames.append("ltx2_stage1_base")
                lora_paths.append(self._stage1_lora_path)
                lora_strengths.append(self._stage1_lora_scale)
                lora_targets.append("transformer")
            lora_nicknames.append("ltx2_stage2_distilled")
            lora_paths.append(self._distilled_lora_path)
            lora_strengths.append(1.0)
            lora_targets.append("transformer")
            self.set_lora(
                lora_nickname=lora_nicknames,
                lora_path=lora_paths,
                target=lora_targets,
                strength=lora_strengths,
            )
        else:
            raise ValueError(f"Unknown LTX2 two-stage LoRA phase: {phase}")

        self._active_lora_phase = phase

    def create_pipeline_stages(self, server_args: ServerArgs):
        _add_ltx2_front_stages(self)
        self.add_stage(LTX2HalveResolutionStage())
        self.add_stage(
            LTX2LoRASwitchStage(pipeline=self, phase="stage1"),
            stage_name="ltx2_lora_switch_stage1",
        )
        _add_ltx2_stage1_generation_stages(self)
        self.add_stages(
            [
                LTX2UpsampleStage(
                    spatial_upsampler=self.get_module("spatial_upsampler"),
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                ),
                (
                    LTX2LoRASwitchStage(pipeline=self, phase="stage2"),
                    "ltx2_lora_switch_stage2",
                ),
                LTX2RefinementStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    distilled_sigmas=self.STAGE_2_DISTILLED_SIGMA_VALUES,
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                ),
            ]
        )
        _add_ltx2_decoding_stage(self)


EntryClass = [LTX2Pipeline, LTX2TwoStagePipeline]
