import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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


class LTX2UpsampleStage(PipelineStage):
    """
    Upsample video latents from Stage 1 (half-res) to full resolution using
    a spatial LatentUpsampler. Also re-packs video and audio latents for Stage 2.

    Flow:
      1. Denormalize video latents (using VAE per-channel stats)
      2. Run LatentUpsampler (spatial 2x)
      3. Re-normalize video latents
      4. Double batch.height / batch.width
      5. Re-pack video latents to [B, S', D]
      6. Re-pack + re-normalize audio latents to [B, S, D]
    """

    def __init__(self, spatial_upsampler, vae, audio_vae=None):
        super().__init__()
        self.spatial_upsampler = spatial_upsampler
        self.vae = vae
        self.audio_vae = audio_vae

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        latents = batch.latents  # [B, C, F, H_half, W_half] normalized

        # --- Video: denormalize -> upsample -> renormalize ---
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

        logger.info("Upsampled video latents: %s", list(latents.shape))

        # --- Update resolution (restore to user-requested full resolution) ---
        batch.height = batch.height * 2
        batch.width = batch.width * 2

        # --- Re-pack video latents to [B, S', D] for Stage 2 denoising ---
        batch_size = latents.shape[0]
        latents = server_args.pipeline_config.maybe_pack_latents(
            latents, batch_size, batch
        )
        batch.latents = latents
        batch.raw_latent_shape = latents.shape

        logger.info(
            "Packed video latents for Stage 2: %s (resolution %dx%d)",
            list(latents.shape),
            batch.height,
            batch.width,
        )

        # --- Re-pack + re-normalize audio latents ---
        audio_latents = batch.audio_latents
        if audio_latents is not None and self.audio_vae is not None:
            # After Stage 1 _post_denoising_loop: audio is [B, C, L, M] denormalized
            # Re-pack: [B, C, L, M] -> [B, L, C*M] = [B, S, D]
            audio_packed = audio_latents.transpose(1, 2).flatten(2)

            # Re-normalize using audio VAE stats
            audio_mean = getattr(self.audio_vae, "latents_mean", None)
            audio_std = getattr(self.audio_vae, "latents_std", None)
            if isinstance(audio_mean, torch.Tensor) and isinstance(
                audio_std, torch.Tensor
            ):
                audio_mean = audio_mean.to(
                    device=audio_packed.device, dtype=audio_packed.dtype
                ).view(1, 1, -1)
                audio_std = audio_std.to(
                    device=audio_packed.device, dtype=audio_packed.dtype
                ).view(1, 1, -1)
                audio_packed = (audio_packed - audio_mean) / audio_std

            batch.audio_latents = audio_packed
            batch.raw_audio_latent_shape = audio_packed.shape
            logger.info(
                "Re-packed audio latents for Stage 2: %s", list(audio_packed.shape)
            )

        return batch
