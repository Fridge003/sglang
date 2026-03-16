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
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    LTX2AVDecodingStage,
    LTX2AVDenoisingStage,
    LTX2AVLatentPreparationStage,
    LTX2HalveResolutionStage,
    LTX2RefinementStage,
    LTX2TextConnectorStage,
    LTX2UpsampleStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
    height = batch.height
    width = batch.width
    num_frames = batch.num_frames

    vae_arch = getattr(
        getattr(server_args.pipeline_config, "vae_config", None), "arch_config", None
    )
    vae_scale_factor = (
        getattr(vae_arch, "spatial_compression_ratio", None)
        or getattr(vae_arch, "vae_scale_factor", None)
        or getattr(server_args.pipeline_config, "vae_scale_factor", None)
    )
    vae_temporal_compression = getattr(
        vae_arch, "temporal_compression_ratio", None
    ) or getattr(server_args.pipeline_config, "vae_temporal_compression", None)

    # Values from LTX2Pipeline in diffusers
    mu = calculate_shift(
        4096,
        base_seq_len=1024,
        max_seq_len=4096,
        base_shift=0.95,
        max_shift=2.05,
    )
    return "mu", mu


class LTX2FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """Override ``_time_shift_exponential`` to use torch f32 instead of numpy f64."""

    def _time_shift_exponential(self, mu, sigma, t):
        if isinstance(t, np.ndarray):
            t_torch = torch.from_numpy(t).to(torch.float32)
            result = math.exp(mu) / (math.exp(mu) + (1 / t_torch - 1) ** sigma)
            return result.numpy()
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class LTX2Pipeline(ComposedPipelineBase):
    # NOTE: must match `model_index.json`'s `_class_name` for native dispatch.
    pipeline_name = "LTX2Pipeline"

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

    STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

    def initialize_pipeline(self, server_args: ServerArgs):
        orig = self.get_module("scheduler")
        self.modules["scheduler"] = LTX2FlowMatchScheduler.from_config(orig.config)

        # Load optional spatial_upsampler if path is provided
        upsampler_path = server_args.component_paths.get("spatial_upsampler")
        if upsampler_path is not None:
            logger.info(
                "Loading spatial_upsampler from %s (two-stage mode)", upsampler_path
            )
            module, memory_usage = PipelineComponentLoader.load_component(
                component_name="spatial_upsampler",
                component_model_path=upsampler_path,
                transformers_or_diffusers="diffusers",
                server_args=server_args,
            )
            self.modules["spatial_upsampler"] = module
            self.memory_usages["spatial_upsampler"] = memory_usage
            logger.info("Two-stage mode enabled")
        else:
            self.modules["spatial_upsampler"] = None

    def create_pipeline_stages(self, server_args: ServerArgs):
        spatial_upsampler = self.get_module("spatial_upsampler")
        is_two_stage = spatial_upsampler is not None

        # Shared front stages
        self.add_stages(
            [
                InputValidationStage(),
                TextEncodingStage(
                    text_encoders=[self.get_module("text_encoder")],
                    tokenizers=[self.get_module("tokenizer")],
                ),
                LTX2TextConnectorStage(connectors=self.get_module("connectors")),
            ]
        )

        if is_two_stage:
            self.add_stage(LTX2HalveResolutionStage())

        self.add_standard_timestep_preparation_stage(prepare_extra_kwargs=[prepare_mu])

        self.add_stages(
            [
                LTX2AVLatentPreparationStage(
                    scheduler=self.get_module("scheduler"),
                    transformer=self.get_module("transformer"),
                    audio_vae=self.get_module("audio_vae"),
                ),
                LTX2AVDenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                ),
            ]
        )

        if is_two_stage:
            self.add_stages(
                [
                    LTX2UpsampleStage(
                        spatial_upsampler=spatial_upsampler,
                        vae=self.get_module("vae"),
                        audio_vae=self.get_module("audio_vae"),
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

        self.add_stage(
            LTX2AVDecodingStage(
                vae=self.get_module("vae"),
                audio_vae=self.get_module("audio_vae"),
                vocoder=self.get_module("vocoder"),
                pipeline=self,
            )
        )


EntryClass = LTX2Pipeline
