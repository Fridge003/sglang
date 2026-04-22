import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.probe_utils import dump_probe_payload

logger = init_logger(__name__)


class LTX2HalveResolutionStage(PipelineStage):
    """Halve batch height/width for two-stage Stage 1 (low-res generation)."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        original_h, original_w = batch.height, batch.width

        vae_scale_factor = getattr(server_args.pipeline_config, "vae_scale_factor", 32)
        required_alignment = max(64, int(vae_scale_factor) * 2)
        if original_h % required_alignment != 0 or original_w % required_alignment != 0:
            raise ValueError(
                "LTX-2 two-stage requires resolution divisible by "
                f"{required_alignment}, got ({original_h}x{original_w})."
            )

        batch.height = batch.height // 2
        batch.width = batch.width // 2
        logger.info(
            "Halved resolution: %dx%d -> %dx%d",
            original_h,
            original_w,
            batch.height,
            batch.width,
        )
        return batch


class LTX2LoRASwitchStage(PipelineStage):
    """Switch LoRA configuration for the requested two-stage phase."""

    def __init__(self, pipeline, phase: str):
        super().__init__()
        self.pipeline = pipeline
        self.phase = phase

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        switch_fn = getattr(self.pipeline, "switch_lora_phase", None)
        should_skip_switch_stage = getattr(
            self.pipeline, "should_skip_ltx2_lora_switch_stage", None
        )
        if callable(should_skip_switch_stage) and should_skip_switch_stage():
            batch.extra["ltx2_phase"] = self.phase
            return batch
        if not callable(switch_fn):
            raise ValueError(
                "LTX2LoRASwitchStage requires pipeline.switch_lora_phase()"
            )
        switch_fn(self.phase, batch=batch)
        batch.extra["ltx2_phase"] = self.phase
        return batch


class LTX2UpsampleStage(PipelineStage):
    """Upsample Stage-1 video latents and prepare Stage-2 inputs."""

    def __init__(self, spatial_upsampler, vae, audio_vae=None, pipeline=None):
        super().__init__()
        self.spatial_upsampler = spatial_upsampler
        self.vae = vae
        self.audio_vae = audio_vae
        self.pipeline = pipeline

    def _upsample_video_latents(
        self, latents: torch.Tensor, server_args: ServerArgs, device: torch.device
    ) -> torch.Tensor:
        vae_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(
            device=device, dtype=latents.dtype
        )
        vae_std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(
            device=device, dtype=latents.dtype
        )
        latents = latents * vae_std + vae_mean
        self.spatial_upsampler = self.spatial_upsampler.to(
            device=device, dtype=latents.dtype
        )
        latents = self.spatial_upsampler(latents)
        if server_args.vae_cpu_offload:
            self.spatial_upsampler = self.spatial_upsampler.to("cpu")
        latents = (latents - vae_mean) / vae_std
        return latents

    @staticmethod
    def _restore_full_resolution(batch: Req) -> None:
        batch.height *= 2
        batch.width *= 2

    @staticmethod
    def _pack_video_latents(
        batch: Req, latents: torch.Tensor, server_args: ServerArgs
    ) -> None:
        batch_size = latents.shape[0]
        latents = server_args.pipeline_config.maybe_pack_latents(
            latents, batch_size, batch
        )
        batch.latents = latents
        batch.raw_latent_shape = latents.shape

    def _repack_audio_latents(self, batch: Req, server_args: ServerArgs) -> None:
        if batch.audio_latents is None or self.audio_vae is None:
            return
        audio_latents = server_args.pipeline_config.maybe_pack_audio_latents(
            batch.audio_latents, batch.audio_latents.shape[0], batch
        )
        batch.audio_latents = audio_latents
        batch.raw_audio_latent_shape = audio_latents.shape
        logger.info(
            "Re-packed audio latents for Stage 2: %s", list(audio_latents.shape)
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        prefetch_stage2 = (
            getattr(self.pipeline, "prefetch_ltx2_stage2_after_stage1", None)
            if self.pipeline is not None
            else None
        )
        if callable(prefetch_stage2):
            prefetch_stage2()

        device = get_local_torch_device()

        if envs.SGLANG_DIFFUSION_PROBE_DIR:
            dump_probe_payload(
                batch,
                "stage1/output",
                {
                    "video_state": {"latent": batch.latents},
                    "audio_state": {"latent": batch.audio_latents},
                },
            )

        inject_path = envs.SGLANG_DIFFUSION_LTX2_INJECT_STAGE1_OUTPUT
        if inject_path:
            print(
                f"[INJECT_STAGE1] loading official stage1 output from {inject_path}",
                flush=True,
            )
            inj = torch.load(str(inject_path), map_location="cpu")
            off_v = inj["video_state"]["latent"]
            off_a = inj["audio_state"]["latent"]
            print(
                f"[INJECT_STAGE1] native pre-inject: video={tuple(batch.latents.shape)} "
                f"audio={tuple(batch.audio_latents.shape) if batch.audio_latents is not None else None}; "
                f"dump: video={tuple(off_v.shape)} audio={tuple(off_a.shape)}",
                flush=True,
            )
            batch.latents = off_v.to(device=batch.latents.device, dtype=batch.latents.dtype)
            if batch.audio_latents is not None:
                batch.audio_latents = off_a.to(
                    device=batch.audio_latents.device, dtype=batch.audio_latents.dtype
                )

        latents = self._upsample_video_latents(batch.latents, server_args, device)
        logger.info("Upsampled video latents: %s", list(latents.shape))

        up_inject = envs.SGLANG_DIFFUSION_LTX2_INJECT_UPSAMPLE_OUTPUT
        if up_inject:
            inj = torch.load(str(up_inject), map_location="cpu")
            off_up = inj["upscaled_video_latent"]
            off_audio = inj["audio_latent_pre_stage2"]
            print(
                f"[INJECT_UPSAMPLE] loading {up_inject}; native_pre: video={tuple(latents.shape)} "
                f"audio={tuple(batch.audio_latents.shape) if batch.audio_latents is not None else None}; "
                f"dump: video={tuple(off_up.shape)} audio={tuple(off_audio.shape)}",
                flush=True,
            )
            latents = off_up.to(device=latents.device, dtype=latents.dtype)
            if batch.audio_latents is not None:
                batch.audio_latents = off_audio.to(
                    device=batch.audio_latents.device, dtype=batch.audio_latents.dtype
                )

        self._restore_full_resolution(batch)
        self._pack_video_latents(batch, latents, server_args)
        logger.info(
            "Packed video latents for Stage 2: %s (resolution %dx%d)",
            list(batch.latents.shape),
            batch.height,
            batch.width,
        )
        self._repack_audio_latents(batch, server_args)
        return batch
