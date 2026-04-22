import copy
from dataclasses import dataclass, field

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    is_ltx2_two_stage_pipeline_name,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.probe_utils import (
    dump_probe_payload,
    get_probe_request_dir,
)

logger = init_logger(__name__)

LTX23_TWO_STAGE_STAGE1_GUIDER_DEFAULTS: dict[str, object] = {
    "video_cfg_scale": 3.0,
    "video_stg_scale": 1.0,
    "video_rescale_scale": 0.7,
    "video_modality_scale": 3.0,
    "video_skip_step": 0,
    "video_stg_blocks": [28],
    "audio_cfg_scale": 7.0,
    "audio_stg_scale": 1.0,
    "audio_rescale_scale": 0.7,
    "audio_modality_scale": 3.0,
    "audio_skip_step": 0,
    "audio_stg_blocks": [28],
}

LTX23_TWO_STAGE_HQ_STAGE1_GUIDER_DEFAULTS: dict[str, object] = {
    "video_cfg_scale": 3.0,
    "video_stg_scale": 0.0,
    "video_rescale_scale": 0.45,
    "video_modality_scale": 3.0,
    "video_skip_step": 0,
    "video_stg_blocks": [],
    "audio_cfg_scale": 7.0,
    "audio_stg_scale": 0.0,
    "audio_rescale_scale": 1.0,
    "audio_modality_scale": 3.0,
    "audio_skip_step": 0,
    "audio_stg_blocks": [],
}

LTX23_RES2S_STEP_NOISE_SEED = -1
LTX23_RES2S_SUBSTEP_NOISE_SEED = 9999


@dataclass(slots=True)
class LTX2DenoisingContext(DenoisingContext):
    """Loop-scoped denoising state for joint LTX-2 video and audio generation."""

    audio_latents: torch.Tensor | None = None
    audio_scheduler: object | None = None
    is_ltx23_variant: bool = False
    use_ltx23_legacy_one_stage: bool = False
    replicate_audio_for_sp: bool = False
    stage: str = "one_stage"
    latent_num_frames_for_model: int = 0
    latent_height: int = 0
    latent_width: int = 0
    denoise_mask: torch.Tensor | None = None
    clean_latent: torch.Tensor | None = None
    last_denoised_video: torch.Tensor | None = None
    last_denoised_audio: torch.Tensor | None = None
    trajectory_audio_latents: list[torch.Tensor] = field(default_factory=list)
    use_native_hq_res2s_sde_noise: bool = False
    res2s_step_noise_generator: torch.Generator | None = None
    res2s_substep_noise_generator: torch.Generator | None = None


class LTX2DenoisingStage(DenoisingStage):
    """
    LTX-2 specific denoising stage that handles joint video and audio generation.
    """

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        *,
        sampler_name: str = "euler",
        **kwargs,
    ):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self.sampler_name = sampler_name

    @staticmethod
    def _get_video_latent_num_frames_for_model(
        batch: Req, server_args: ServerArgs, latents: torch.Tensor
    ) -> int:
        """Return the latent-frame length the DiT model should see.

        - If video latents were time-sharded for SP and are packed as token latents
          ([B, S, D]), the model only sees the local shard and must use the local
          latent-frame count (stored on the batch during SP sharding).
        - Otherwise, fall back to the global latent-frame count inferred from the
          requested output frames and the VAE temporal compression ratio.
        """
        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        is_token_latents = isinstance(latents, torch.Tensor) and latents.ndim == 3

        if did_sp_shard and is_token_latents:
            if not hasattr(batch, "sp_video_latent_num_frames"):
                raise ValueError(
                    "SP-sharded LTX2 token latents require `batch.sp_video_latent_num_frames` "
                    "to be set by `LTX2PipelineConfig.shard_latents_for_sp()`."
                )
            return int(batch.sp_video_latent_num_frames)

        pc = server_args.pipeline_config
        return int(
            (batch.num_frames - 1)
            // int(pc.vae_config.arch_config.temporal_compression_ratio)
            + 1
        )

    @staticmethod
    def _truncate_sp_padded_token_latents(
        batch: Req, latents: torch.Tensor
    ) -> torch.Tensor:
        """Remove token padding introduced by SP time-sharding (if applicable)."""
        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        if not did_sp_shard or not (
            isinstance(latents, torch.Tensor) and latents.ndim == 3
        ):
            return latents

        raw_shape = getattr(batch, "raw_latent_shape", None)
        if not (isinstance(raw_shape, tuple) and len(raw_shape) == 3):
            return latents

        orig_s = int(raw_shape[1])
        cur_s = int(latents.shape[1])
        if cur_s == orig_s:
            return latents
        if cur_s < orig_s:
            raise ValueError(
                f"Unexpected gathered token-latents seq_len {cur_s} < original seq_len {orig_s}."
            )
        return latents[:, :orig_s, :].contiguous()

    def _maybe_enable_cache_dit(self, num_inference_steps: int, batch: Req) -> None:
        """Disable cache-dit for TI2V-style requests (image-conditioned), to avoid stale activations.

        NOTE: base denoising stage calls this hook with (num_inference_steps, batch).
        """
        if getattr(self, "_disable_cache_dit_for_request", False):
            return
        return super()._maybe_enable_cache_dit(num_inference_steps, batch)

    def _get_ltx2_stage1_guider_params(
        self, batch: Req, server_args: ServerArgs, stage: str
    ) -> dict[str, object] | None:
        if stage != "stage1":
            return None
        request_params = batch.extra.get("ltx2_stage1_guider_params")
        if not is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
            return request_params
        default_params = copy.deepcopy(
            LTX23_TWO_STAGE_HQ_STAGE1_GUIDER_DEFAULTS
            if server_args.pipeline_class_name == "LTX2TwoStageHQPipeline"
            else LTX23_TWO_STAGE_STAGE1_GUIDER_DEFAULTS
        )
        if request_params is None:
            return default_params
        for key, value in request_params.items():
            if value is None:
                continue
            default_params[key] = list(value) if isinstance(value, list) else value
        return default_params

    @staticmethod
    def _randn_like_with_batch_generators(
        reference_tensor: torch.Tensor, batch: Req
    ) -> torch.Tensor:
        generator = getattr(batch, "generator", None)
        if isinstance(generator, list):
            bsz = int(reference_tensor.shape[0])
            valid_generators = [g for g in generator if isinstance(g, torch.Generator)]
            if len(valid_generators) == 1:
                generator = valid_generators[0]
            elif len(valid_generators) >= bsz:
                generator = valid_generators[:bsz]
            else:
                generator = None
        elif not isinstance(generator, torch.Generator):
            generator = None

        return randn_tensor(
            reference_tensor.shape,
            generator=generator,
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )

    @staticmethod
    def _ltx2_channelwise_normalize(noise: torch.Tensor) -> torch.Tensor:
        return noise.sub_(
            noise.mean(dim=(-2, -1), keepdim=True)
        ).div_(noise.std(dim=(-2, -1), keepdim=True))

    @classmethod
    def _ltx2_res2s_new_noise(
        cls,
        reference_tensor: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = torch.randn(
            reference_tensor.shape,
            generator=generator,
            dtype=torch.float64,
            device=reference_tensor.device,
        )
        noise = (noise - noise.mean()) / noise.std()
        return cls._ltx2_channelwise_normalize(noise)

    @staticmethod
    def _ltx2_init_res2s_noise_generators(ctx: LTX2DenoisingContext) -> None:
        reference_tensor = (
            ctx.latents if isinstance(ctx.latents, torch.Tensor) else ctx.audio_latents
        )
        if reference_tensor is None:
            raise ValueError("LTX-2 res2s requires video or audio latents.")
        device = reference_tensor.device
        ctx.res2s_step_noise_generator = torch.Generator(device=device).manual_seed(
            LTX23_RES2S_STEP_NOISE_SEED
        )
        ctx.res2s_substep_noise_generator = torch.Generator(device=device).manual_seed(
            LTX23_RES2S_SUBSTEP_NOISE_SEED
        )

    @classmethod
    def _ltx2_res2s_noise_like(
        cls,
        reference_tensor: torch.Tensor,
        ctx: LTX2DenoisingContext,
        *,
        substep: bool,
    ) -> torch.Tensor:
        generator = (
            ctx.res2s_substep_noise_generator
            if substep
            else ctx.res2s_step_noise_generator
        )
        if generator is None:
            raise ValueError("LTX-2 res2s noise generator was not initialized.")
        return cls._ltx2_res2s_new_noise(reference_tensor, generator).to(
            dtype=reference_tensor.dtype
        )

    @staticmethod
    def _ltx2_should_skip_step(step_index: int, skip_step: int) -> bool:
        if skip_step == 0:
            return False
        return step_index % (skip_step + 1) != 0

    @staticmethod
    def _ltx2_apply_rescale(
        cond: torch.Tensor, pred: torch.Tensor, rescale_scale: float
    ) -> torch.Tensor:
        if rescale_scale == 0.0:
            return pred
        factor = cond.std() / pred.std()
        factor = rescale_scale * factor + (1.0 - rescale_scale)
        return pred * factor

    @staticmethod
    def _should_dump_phase_probe_once(
        batch: Req, probe_key: str, *, phase: str | None = None
    ) -> bool:
        phase = phase or str(batch.extra.get("ltx2_phase") or "")
        if not phase:
            return False
        probe_state_key = f"_ltx2_{phase}_{probe_key}_dumped"
        if batch.extra.get(probe_state_key):
            return False
        batch.extra[probe_state_key] = True
        return True

    @staticmethod
    def _maybe_dump_phase_step0_probe(
        batch: Req,
        *,
        phase: str,
        step_index: int,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        latent_model_input: torch.Tensor,
        audio_latent_model_input: torch.Tensor,
        timestep_video: torch.Tensor | None,
        timestep_audio: torch.Tensor | None,
        prompt_timestep_video: torch.Tensor | None,
        prompt_timestep_audio: torch.Tensor | None,
        model_video: torch.Tensor | None = None,
        model_audio: torch.Tensor | None = None,
        latents_after_step: torch.Tensor | None = None,
        audio_latents_after_step: torch.Tensor | None = None,
    ) -> None:
        if step_index != 0:
            return
        if not LTX2DenoisingStage._should_dump_phase_probe_once(
            batch, "step0_probe", phase=phase
        ):
            return
        dump_probe_payload(
            batch,
            f"denoising/{phase}_step0",
            {
                "encoder_hidden_states": encoder_hidden_states,
                "audio_encoder_hidden_states": audio_encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
                "latent_model_input": latent_model_input,
                "audio_latent_model_input": audio_latent_model_input,
                "timestep_video": timestep_video,
                "timestep_audio": timestep_audio,
                "prompt_timestep_video": prompt_timestep_video,
                "prompt_timestep_audio": prompt_timestep_audio,
                "model_video": model_video,
                "model_audio": model_audio,
                "latents_after_step": latents_after_step,
                "audio_latents_after_step": audio_latents_after_step,
            },
        )

    @staticmethod
    def _maybe_dump_phase_transformer_step0_probe(
        batch: Req,
        *,
        phase: str,
        step_index: int,
        payload: dict[str, object],
    ) -> None:
        if step_index != 0:
            return
        if not LTX2DenoisingStage._should_dump_phase_probe_once(
            batch, "transformer_step0_probe", phase=phase
        ):
            return
        dump_probe_payload(batch, f"denoising/{phase}_transformer_step0", payload)

    @staticmethod
    def _ltx2_apply_clean_latent_mask(
        latents: torch.Tensor,
        ctx: LTX2DenoisingContext,
    ) -> torch.Tensor:
        if ctx.denoise_mask is None or ctx.clean_latent is None:
            return latents
        return (
            latents.float() * ctx.denoise_mask
            + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
        ).to(dtype=latents.dtype)

    @staticmethod
    def _ltx2_phi_1(neg_h: torch.Tensor) -> torch.Tensor:
        small = neg_h.abs() < 1e-4
        series = 1.0 + 0.5 * neg_h + (neg_h * neg_h) / 6.0
        return torch.where(small, series, torch.expm1(neg_h) / neg_h)

    @classmethod
    def _ltx2_phi_2(cls, neg_h: torch.Tensor) -> torch.Tensor:
        small = neg_h.abs() < 1e-4
        series = 0.5 + neg_h / 6.0 + (neg_h * neg_h) / 24.0
        exact = (torch.expm1(neg_h) - neg_h) / (neg_h * neg_h)
        return torch.where(small, series, exact)

    @classmethod
    def _ltx2_get_res2s_coefficients(
        cls, h: torch.Tensor, c2: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a21 = c2 * cls._ltx2_phi_1(-h * c2)
        b2 = cls._ltx2_phi_2(-h) / c2
        b1 = cls._ltx2_phi_1(-h) - b2
        return a21, b1, b2

    @staticmethod
    def _ltx2_get_sde_coeff(
        sigma_next: torch.Tensor,
        *,
        sigma_up: torch.Tensor | None = None,
        sigma_down: torch.Tensor | None = None,
        sigma_max: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sigma_down is not None:
            alpha_ratio = (1.0 - sigma_next) / (1.0 - sigma_down)
            sigma_up = torch.sqrt(
                torch.clamp(
                    sigma_next.square() - sigma_down.square() * alpha_ratio.square(),
                    min=0.0,
                )
            )
        elif sigma_up is not None:
            sigma_up = torch.minimum(sigma_up, sigma_next * 0.9999)
            sigmax = sigma_max if sigma_max is not None else torch.ones_like(sigma_next)
            sigma_signal = sigmax - sigma_next
            sigma_residual = torch.sqrt(
                torch.clamp(sigma_next.square() - sigma_up.square(), min=0.0)
            )
            alpha_ratio = sigma_signal + sigma_residual
            sigma_down = sigma_residual / alpha_ratio
        else:
            alpha_ratio = torch.ones_like(sigma_next)
            sigma_down = sigma_next
            sigma_up = torch.zeros_like(sigma_next)
        return (
            torch.nan_to_num(alpha_ratio),
            torch.nan_to_num(sigma_down),
            torch.nan_to_num(sigma_up),
        )

    @classmethod
    def _ltx2_res2s_sde_step(
        cls,
        *,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        noise: torch.Tensor,
        eta: float = 0.5,
    ) -> torch.Tensor:
        alpha_ratio, sigma_down, sigma_up = cls._ltx2_get_sde_coeff(
            sigma_next,
            sigma_up=sigma_next * eta,
        )
        if bool((sigma_up == 0).any()) or bool((sigma_next == 0).any()):
            return denoised_sample.to(dtype=sample.dtype)
        eps_next = (sample - denoised_sample) / (sigma - sigma_next)
        denoised_next = sample - sigma * eps_next
        x_noised = (
            alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise
        )
        return x_noised.to(dtype=sample.dtype)

    def _ltx2_stage2_res2s_step(
        self,
        *,
        ctx: "LTX2DenoisingContext",
        batch: Req,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        model_video_velocity: torch.Tensor,
        model_audio_velocity: torch.Tensor,
        midpoint_model_call,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """res2s RK2 step for unguided stage-2 refinement (no CFG/STG).

        Converts the velocity predictions to x_0 denoised estimates and runs the
        official res2s update: midpoint SDE, optional bongmath anchor refinement,
        midpoint denoiser re-eval, then the final RK2 combination with SDE noise.
        Mirrors the math used in the guided stage-1 branch further down in
        ``_run_denoising_step`` but skips the guider and works from raw model
        velocities.
        """
        sigma_val = float(sigma.item())
        sigma_next_val = float(sigma_next.item())

        # Convert velocity -> x_0 (denoised). `model_video_velocity` matches
        # `(x_t - x_0) / sigma` in LTX flow-match, so `x_0 = x_t - sigma * v`.
        if sigma_val == 0.0:
            denoised_video = ctx.latents.float()
            denoised_audio = ctx.audio_latents.float()
        else:
            denoised_video = (
                ctx.latents.float() - sigma * model_video_velocity.float()
            )
            denoised_audio = (
                ctx.audio_latents.float() - sigma * model_audio_velocity.float()
            )

        if sigma_val == 0.0 or sigma_next_val == 0.0:
            next_video = denoised_video.to(dtype=ctx.latents.dtype)
            next_audio = denoised_audio.to(dtype=ctx.audio_latents.dtype)
            next_video = self._ltx2_apply_clean_latent_mask(next_video, ctx)
            return next_video, next_audio

        sigma_d = sigma.double()
        sigma_next_d = sigma_next.double()
        h = -torch.log(torch.clamp(sigma_next_d / sigma_d, min=1e-12))
        a21, b1, b2 = self._ltx2_get_res2s_coefficients(h)
        sub_sigma = torch.sqrt(torch.clamp(sigma_d * sigma_next_d, min=0.0))

        anchor_video = ctx.latents.double()
        anchor_audio = ctx.audio_latents.double()
        eps1_video = denoised_video.double() - anchor_video
        eps1_audio = denoised_audio.double() - anchor_audio

        midpoint_video_det = anchor_video + h * a21 * eps1_video
        midpoint_audio_det = anchor_audio + h * a21 * eps1_audio

        midpoint_video_latents = self._ltx2_res2s_sde_step(
            sample=anchor_video,
            denoised_sample=midpoint_video_det,
            sigma=sigma_d,
            sigma_next=sub_sigma,
            noise=(
                self._ltx2_res2s_noise_like(
                    ctx.latents, ctx, substep=True
                ).float()
                if ctx.use_native_hq_res2s_sde_noise
                else self._randn_like_with_batch_generators(
                    ctx.latents, batch
                ).float()
            ),
        )
        midpoint_audio_latents = self._ltx2_res2s_sde_step(
            sample=anchor_audio,
            denoised_sample=midpoint_audio_det,
            sigma=sigma_d,
            sigma_next=sub_sigma,
            noise=(
                self._ltx2_res2s_noise_like(
                    ctx.audio_latents, ctx, substep=True
                ).float()
                if ctx.use_native_hq_res2s_sde_noise
                else self._randn_like_with_batch_generators(
                    ctx.audio_latents, batch
                ).float()
            ),
        )
        midpoint_video_latents = self._ltx2_apply_clean_latent_mask(
            midpoint_video_latents.to(dtype=ctx.latents.dtype), ctx
        )
        midpoint_audio_latents = midpoint_audio_latents.to(
            dtype=ctx.audio_latents.dtype
        )

        # Bongmath anchor refinement (h < 0.5 and sigma > 0.03 -> first stage-2 step).
        if float(h.item()) < 0.5 and sigma_val > 0.03:
            x_mid_v = midpoint_video_latents.double()
            x_mid_a = midpoint_audio_latents.double()
            for _ in range(100):
                anchor_video = x_mid_v - h * a21 * eps1_video
                eps1_video = denoised_video.double() - anchor_video
                anchor_audio = x_mid_a - h * a21 * eps1_audio
                eps1_audio = denoised_audio.double() - anchor_audio

        mid_v_velocity, mid_a_velocity = midpoint_model_call(
            midpoint_video_latents, midpoint_audio_latents, sub_sigma
        )
        midpoint_denoised_video = (
            midpoint_video_latents.float() - sub_sigma * mid_v_velocity
        )
        midpoint_denoised_audio = (
            midpoint_audio_latents.float() - sub_sigma * mid_a_velocity
        )

        eps2_video = midpoint_denoised_video.double() - anchor_video
        eps2_audio = midpoint_denoised_audio.double() - anchor_audio

        next_video_det = anchor_video + h * (b1 * eps1_video + b2 * eps2_video)
        next_audio_det = anchor_audio + h * (b1 * eps1_audio + b2 * eps2_audio)

        next_video = self._ltx2_res2s_sde_step(
            sample=anchor_video,
            denoised_sample=next_video_det,
            sigma=sigma_d,
            sigma_next=sigma_next_d,
            noise=(
                self._ltx2_res2s_noise_like(
                    ctx.latents, ctx, substep=False
                ).float()
                if ctx.use_native_hq_res2s_sde_noise
                else self._randn_like_with_batch_generators(
                    ctx.latents, batch
                ).float()
            ),
        )
        next_audio = self._ltx2_res2s_sde_step(
            sample=anchor_audio,
            denoised_sample=next_audio_det,
            sigma=sigma_d,
            sigma_next=sigma_next_d,
            noise=(
                self._ltx2_res2s_noise_like(
                    ctx.audio_latents, ctx, substep=False
                ).float()
                if ctx.use_native_hq_res2s_sde_noise
                else self._randn_like_with_batch_generators(
                    ctx.audio_latents, batch
                ).float()
            ),
        )
        next_video = self._ltx2_apply_clean_latent_mask(
            next_video.to(dtype=ctx.latents.dtype), ctx
        )
        next_audio = next_audio.to(dtype=ctx.audio_latents.dtype)
        return next_video, next_audio

    @staticmethod
    def _prepare_ltx2_ti2v_clean_state(
        latents: torch.Tensor,
        image_latent: torch.Tensor,
        num_img_tokens: int,
        zero_clean_latent: bool,
        clean_latent_background: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = latents.clone()
        conditioned = image_latent[:, :num_img_tokens, :].to(
            device=latents.device, dtype=latents.dtype
        )
        latents[:, :num_img_tokens, :] = conditioned
        denoise_mask = torch.ones(
            (latents.shape[0], latents.shape[1], 1),
            device=latents.device,
            dtype=torch.float32,
        )
        denoise_mask[:, :num_img_tokens, :] = 0.0
        if clean_latent_background is not None:
            clean_latent = (
                clean_latent_background.detach()
                .clone()
                .to(device=latents.device, dtype=latents.dtype)
            )
        elif zero_clean_latent:
            clean_latent = torch.zeros_like(latents)
        else:
            clean_latent = latents.detach().clone()
        clean_latent[:, :num_img_tokens, :] = conditioned
        return latents, denoise_mask, clean_latent

    @staticmethod
    def _ltx2_velocity_to_x0(
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device=sample.device, dtype=torch.float32)
            while sigma.ndim < sample.ndim:
                sigma = sigma.unsqueeze(-1)
            return (sample.float() - sigma * velocity.float()).to(sample.dtype)
        return (sample.float() - float(sigma) * velocity.float()).to(sample.dtype)

    @staticmethod
    def _repeat_batch_dim(tensor: torch.Tensor, target_batch_size: int) -> torch.Tensor:
        """Repeat along batch dim while preserving any tokenwise timestep layout."""
        if tensor.shape[0] == int(target_batch_size):
            return tensor
        if tensor.shape[0] <= 0 or int(target_batch_size) % int(tensor.shape[0]) != 0:
            raise ValueError(
                f"Cannot repeat tensor with batch={tensor.shape[0]} to target_batch_size={target_batch_size}"
            )
        repeat_factor = int(target_batch_size) // int(tensor.shape[0])
        return tensor.repeat(repeat_factor, *([1] * (tensor.ndim - 1)))

    @staticmethod
    def _build_ltx2_sp_padding_mask(
        batch: Req,
        *,
        seq_len: int,
        batch_size: int,
        key: str,
        device: torch.device,
    ) -> torch.Tensor | None:
        valid = getattr(batch, key, None)
        if valid is None:
            return None
        valid = int(valid)
        if valid <= 0 or valid >= int(seq_len):
            return None
        mask = torch.ones(
            (batch_size, int(seq_len)), device=device, dtype=torch.float32
        )
        mask[:, valid:] = 0.0
        return mask

    @staticmethod
    def _get_ltx_prompt_attention_mask(
        batch: Req,
        *,
        is_ltx23_variant: bool,
        negative: bool = False,
    ) -> torch.Tensor | None:
        if is_ltx23_variant:
            return None
        return (
            batch.negative_attention_mask if negative else batch.prompt_attention_mask
        )

    @classmethod
    def _should_use_ltx23_legacy_one_stage(
        cls,
        server_args: ServerArgs,
        pipeline_name: str | None,
    ) -> bool:
        if not is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        ):
            return False
        if is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
            return False
        return not is_ltx2_two_stage_pipeline_name(pipeline_name)

    @classmethod
    def _should_shard_ltx23_legacy_one_stage_audio_latents(
        cls,
        batch: Req,
        server_args: ServerArgs,
    ) -> bool:
        return bool(
            get_sp_world_size() > 1
            and is_ltx23_native_variant(
                server_args.pipeline_config.vae_config.arch_config
            )
            and cls._should_use_ltx23_legacy_one_stage(server_args, None)
            and server_args.pipeline_config.can_shard_audio_latents_for_sp(
                batch.audio_latents
            )
        )

    @staticmethod
    def _should_use_ltx23_two_stage_cfg_pair_batch(
        batch: Req,
        ctx: LTX2DenoisingContext,
        server_args: ServerArgs,
    ) -> bool:
        if not is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
            return False
        if not ctx.is_ltx23_variant or ctx.use_ltx23_legacy_one_stage:
            return False
        if int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0:
            return False
        if get_sp_world_size() != 1:
            return False
        if int(server_args.num_gpus) != 1:
            return False
        return int(server_args.tp_size or 1) == 1

    @staticmethod
    def _should_use_split_two_stage_guided_passes(
        batch: Req,
        server_args: ServerArgs,
    ) -> bool:
        return (
            is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name)
            and int(server_args.num_gpus) == 1
            and int(server_args.tp_size or 1) == 1
            and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0
        )

    @classmethod
    def _ltx2_calculate_guided_x0(
        cls,
        *,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
        cfg_scale: float,
        stg_scale: float,
        rescale_scale: float,
        modality_scale: float,
    ) -> torch.Tensor:
        pred = (
            cond
            + (cfg_scale - 1.0) * (cond - uncond_text)
            + stg_scale * (cond - uncond_perturbed)
            + (modality_scale - 1.0) * (cond - uncond_modality)
        )
        return cls._ltx2_apply_rescale(cond, pred, rescale_scale)

    def _preprocess_sp_latents(self, batch: Req, server_args: ServerArgs):
        """LTX-2 TI2V applies image_latent in token space *after* SP sharding,
        so the base implementation must not shard it."""
        saved = batch.image_latent
        batch.image_latent = None
        super()._preprocess_sp_latents(batch, server_args)
        batch.image_latent = saved

    @staticmethod
    def _should_apply_ltx2_ti2v(batch: Req) -> bool:
        """True if we have an image-latent token prefix to condition with.

        SP note: when token latents are time-sharded, only the rank that owns the
        *global* first latent frame should apply TI2V conditioning (rank with start_frame==0).
        """
        if (
            batch.image_latent is None
            or int(getattr(batch, "ltx2_num_image_tokens", 0)) <= 0
        ):
            return False
        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        if not did_sp_shard:
            return True
        return int(getattr(batch, "sp_video_start_frame", 0)) == 0

    @staticmethod
    def _should_replicate_ltx23_audio_for_sp(
        batch: Req,
        server_args: ServerArgs,
        *,
        is_ltx23_variant: bool,
    ) -> bool:
        return False

    @staticmethod
    def _should_use_native_hq_res2s_sde_noise(server_args: ServerArgs) -> bool:
        return server_args.pipeline_class_name == "LTX2TwoStageHQPipeline"

    def _prepare_denoising_loop(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> LTX2DenoisingContext:
        """Extend the base context with LTX-2 audio, SP, and TI2V state."""
        self._disable_cache_dit_for_request = batch.image_path is not None
        base_ctx = super()._prepare_denoising_loop(batch, server_args)
        ctx = LTX2DenoisingContext(**base_ctx.to_kwargs())
        ctx.is_ltx23_variant = is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        )
        phase = batch.extra.get("ltx2_phase")
        pipeline = self.pipeline() if self.pipeline else None
        pipeline_name = pipeline.pipeline_name if pipeline is not None else None
        ctx.use_ltx23_legacy_one_stage = self._should_use_ltx23_legacy_one_stage(
            server_args, pipeline_name
        )
        ctx.use_native_hq_res2s_sde_noise = (
            ctx.is_ltx23_variant
            and self._should_use_native_hq_res2s_sde_noise(server_args)
        )
        ctx.stage = (
            phase
            if phase is not None
            else ("stage1" if ctx.use_ltx23_legacy_one_stage else "one_stage")
        )
        ctx.audio_latents = batch.audio_latents
        # Video and audio keep separate scheduler state throughout the denoising loop.
        ctx.audio_scheduler = copy.deepcopy(self.scheduler)

        do_ti2v = self._should_apply_ltx2_ti2v(batch)

        if ctx.use_ltx23_legacy_one_stage:
            batch.ltx23_audio_replicated_for_sp = False
            batch.did_sp_shard_audio_latents = False
        else:
            ctx.replicate_audio_for_sp = self._should_replicate_ltx23_audio_for_sp(
                batch,
                server_args,
                is_ltx23_variant=ctx.is_ltx23_variant,
            )
            batch.ltx23_audio_replicated_for_sp = bool(ctx.replicate_audio_for_sp)
            if (
                ctx.is_ltx23_variant
                and get_sp_world_size() > 1
                and server_args.pipeline_config.can_shard_audio_latents_for_sp(
                    batch.audio_latents
                )
                and not ctx.replicate_audio_for_sp
            ):
                (
                    batch.audio_latents,
                    batch.did_sp_shard_audio_latents,
                ) = server_args.pipeline_config.shard_audio_latents_for_sp(
                    batch, batch.audio_latents
                )
                ctx.audio_latents = batch.audio_latents
            else:
                batch.did_sp_shard_audio_latents = False

        # For LTX-2 packed token latents, SP sharding happens on the time dimension
        # (frames). The model must see local latent frames (RoPE offset is applied
        # inside the model using SP rank).
        ctx.latent_num_frames_for_model = self._get_video_latent_num_frames_for_model(
            batch=batch, server_args=server_args, latents=ctx.latents
        )
        ctx.latent_height = (
            batch.height
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        ctx.latent_width = (
            batch.width
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        if do_ti2v:
            if not (isinstance(ctx.latents, torch.Tensor) and ctx.latents.ndim == 3):
                raise ValueError("LTX-2 TI2V expects packed token latents [B, S, D].")
            clean_latent_background = getattr(
                batch, "ltx2_ti2v_clean_latent_background", None
            )
            if not (
                isinstance(clean_latent_background, torch.Tensor)
                and clean_latent_background.shape == ctx.latents.shape
            ):
                clean_latent_background = None
            # Keep conditioned tokens clean and reuse the mask during every step update.
            ctx.latents, ctx.denoise_mask, ctx.clean_latent = (
                self._prepare_ltx2_ti2v_clean_state(
                    latents=ctx.latents,
                    image_latent=batch.image_latent,
                    num_img_tokens=int(getattr(batch, "ltx2_num_image_tokens", 0)),
                    zero_clean_latent=ctx.is_ltx23_variant,
                    clean_latent_background=clean_latent_background,
                )
            )
        return ctx

    def _before_denoising_loop(
        self, ctx: LTX2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Reset the mirrored audio scheduler before the shared loop begins."""
        if ctx.stage in ("stage1", "stage2"):
            pipeline = self.pipeline() if self.pipeline else None
            switch_lora_phase = (
                getattr(pipeline, "switch_lora_phase", None)
                if pipeline is not None
                else None
            )
            if callable(switch_lora_phase):
                switch_lora_phase(ctx.stage)
            ensure_phase_ready = (
                getattr(pipeline, "ensure_ltx2_phase_ready", None)
                if pipeline is not None
                else None
            )
            if callable(ensure_phase_ready):
                ensure_phase_ready(ctx.stage)
        super()._before_denoising_loop(ctx, batch, server_args)
        if ctx.audio_scheduler is None:
            raise ValueError("LTX-2 audio scheduler was not prepared.")
        ctx.audio_scheduler.set_begin_index(0)
        if self.sampler_name == "res2s" and ctx.use_native_hq_res2s_sde_noise:
            self._ltx2_init_res2s_noise_generators(ctx)

    def _prepare_step_attn_metadata(
        self,
        ctx: LTX2DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_int: int,
        timesteps_cpu: torch.Tensor,
    ):
        """Preserve the legacy LTX-2 attention-metadata contract."""
        # Legacy LTX-2 paths used the plain attention-metadata builder call here.
        del ctx, t_int, timesteps_cpu
        return self._build_attn_metadata(step_index, batch, server_args)

    def _run_denoising_step(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Run one joint video/audio denoising step with LTX-2-specific guidance."""
        if ctx.audio_latents is None:
            raise ValueError("LTX-2 requires audio latents for denoising.")
        if ctx.audio_scheduler is None:
            raise ValueError("LTX-2 audio scheduler was not prepared.")

        # 1. Read the scheduler sigma pair and derive the Euler delta.
        sigmas = getattr(self.scheduler, "sigmas", None)
        if sigmas is None or not isinstance(sigmas, torch.Tensor):
            raise ValueError("Expected scheduler.sigmas to be a tensor for LTX-2.")
        sigma = sigmas[step.step_index].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        sigma_next = sigmas[step.step_index + 1].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        dt = sigma_next - sigma

        # 2. Materialize the current video/audio latent inputs in the compute dtype.
        latent_model_input = ctx.latents.to(ctx.target_dtype)
        audio_latent_model_input = ctx.audio_latents.to(ctx.target_dtype)
        stage1_guider_params = self._get_ltx2_stage1_guider_params(
            batch, server_args, ctx.stage
        )

        if audio_latent_model_input.ndim == 3:
            audio_num_frames_latent = int(audio_latent_model_input.shape[1])
        elif audio_latent_model_input.ndim == 4:
            audio_num_frames_latent = int(audio_latent_model_input.shape[2])
        else:
            raise ValueError(
                f"Unexpected audio latents rank: {audio_latent_model_input.ndim}, shape={tuple(audio_latent_model_input.shape)}"
            )

        # 3. Prepare any LTX-specific RoPE coordinates and timestep layouts.
        video_coords = None
        audio_coords = None
        if not ctx.use_ltx23_legacy_one_stage:
            video_coords = server_args.pipeline_config.prepare_video_rope_coords_for_sp(
                step.current_model,
                batch,
                latent_model_input,
                num_frames=ctx.latent_num_frames_for_model,
                height=ctx.latent_height,
                width=ctx.latent_width,
            )
            audio_coords = server_args.pipeline_config.prepare_audio_rope_coords_for_sp(
                step.current_model,
                batch,
                audio_latent_model_input,
                num_frames=audio_num_frames_latent,
            )

        batch_size = int(latent_model_input.shape[0])
        probe_request_dir = get_probe_request_dir(batch)
        internal_probe_path = None
        phase = str(batch.extra.get("ltx2_phase") or "")
        if step.step_index == 0 and phase in {"stage1", "stage2"} and probe_request_dir is not None:
            internal_probe_path = str(
                probe_request_dir / f"denoising/{phase}_transformer_preprocessed.pt"
            )
        use_raw_sigma_timestep = (
            ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
        )
        timestep = (
            sigma.to(device=ctx.latents.device, dtype=torch.float32).expand(batch_size)
            if use_raw_sigma_timestep
            else step.t_device.to(device=ctx.latents.device, dtype=torch.float32).expand(
                batch_size
            )
        )
        if ctx.denoise_mask is not None:
            if use_raw_sigma_timestep:
                timestep_video = timestep.view(
                    batch_size, *([1] * (ctx.denoise_mask.ndim - 1))
                ) * ctx.denoise_mask
            else:
                timestep_video = timestep.unsqueeze(-1) * ctx.denoise_mask.squeeze(-1)
        elif use_raw_sigma_timestep:
            timestep_video = timestep.view(batch_size, 1, 1).expand(
                batch_size, int(latent_model_input.shape[1]), 1
            )
        else:
            timestep_video = timestep

        if use_raw_sigma_timestep and audio_latent_model_input.ndim == 3:
            timestep_audio = timestep.view(batch_size, 1, 1).expand(
                batch_size, int(audio_latent_model_input.shape[1]), 1
            )
        else:
            timestep_audio = timestep

        prompt_timestep_video = None
        prompt_timestep_audio = None
        if ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage:
            prompt_timestep_video = sigma.to(
                device=ctx.latents.device, dtype=torch.float32
            ).expand(batch_size)
            prompt_timestep_audio = sigma.to(
                device=ctx.audio_latents.device, dtype=torch.float32
            ).expand(batch_size)

        # 4. Build attention masks that account for SP padding and replicated audio.
        if ctx.use_ltx23_legacy_one_stage:
            video_self_attention_mask = None
            audio_self_attention_mask = None
            a2v_cross_attention_mask = None
            v2a_cross_attention_mask = None
        else:
            video_self_attention_mask = self._build_ltx2_sp_padding_mask(
                batch,
                seq_len=int(latent_model_input.shape[1]),
                batch_size=batch_size,
                key="sp_video_valid_token_count",
                device=latent_model_input.device,
            )
            audio_self_attention_mask = self._build_ltx2_sp_padding_mask(
                batch,
                seq_len=audio_num_frames_latent,
                batch_size=batch_size,
                key="sp_audio_valid_token_count",
                device=audio_latent_model_input.device,
            )
            a2v_cross_attention_mask = audio_self_attention_mask
            v2a_cross_attention_mask = video_self_attention_mask

        def build_model_kwargs(
            *,
            encoder_hidden_states: torch.Tensor,
            audio_encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor | None,
            hidden_states: torch.Tensor | None = None,
            audio_hidden_states: torch.Tensor | None = None,
            timestep_video_input: torch.Tensor | None = None,
            timestep_audio_input: torch.Tensor | None = None,
            prompt_timestep_video_input: torch.Tensor | None = None,
            prompt_timestep_audio_input: torch.Tensor | None = None,
            audio_num_frames_input: int | None = None,
            video_coords_input: torch.Tensor | None = None,
            audio_coords_input: torch.Tensor | None = None,
            video_self_attention_mask_input: torch.Tensor | None = None,
            audio_self_attention_mask_input: torch.Tensor | None = None,
            a2v_cross_attention_mask_input: torch.Tensor | None = None,
            v2a_cross_attention_mask_input: torch.Tensor | None = None,
            skip_video_self_attn_blocks: tuple[int, ...] | None = None,
            skip_audio_self_attn_blocks: tuple[int, ...] | None = None,
            disable_a2v_cross_attn: bool = False,
            disable_v2a_cross_attn: bool = False,
        ) -> dict[str, object]:
            kwargs: dict[str, object] = {
                "hidden_states": (
                    latent_model_input if hidden_states is None else hidden_states
                ),
                "audio_hidden_states": (
                    audio_latent_model_input
                    if audio_hidden_states is None
                    else audio_hidden_states
                ),
                "encoder_hidden_states": encoder_hidden_states,
                "audio_encoder_hidden_states": audio_encoder_hidden_states,
                "timestep": (
                    timestep_video
                    if timestep_video_input is None
                    else timestep_video_input
                ),
                "audio_timestep": (
                    timestep_audio
                    if timestep_audio_input is None
                    else timestep_audio_input
                ),
                "encoder_attention_mask": encoder_attention_mask,
                "audio_encoder_attention_mask": encoder_attention_mask,
                "num_frames": ctx.latent_num_frames_for_model,
                "height": ctx.latent_height,
                "width": ctx.latent_width,
                "fps": batch.fps,
                "audio_num_frames": (
                    audio_num_frames_latent
                    if audio_num_frames_input is None
                    else audio_num_frames_input
                ),
                "video_coords": (
                    video_coords if video_coords_input is None else video_coords_input
                ),
                "audio_coords": (
                    audio_coords if audio_coords_input is None else audio_coords_input
                ),
                "return_latents": False,
                "return_dict": False,
            }
            if not ctx.use_ltx23_legacy_one_stage:
                kwargs.update(
                    {
                        "_sglang_internal_probe_path": internal_probe_path,
                        "_sglang_internal_probe_phase": phase,
                        "prompt_timestep": (
                            prompt_timestep_video
                            if prompt_timestep_video_input is None
                            else prompt_timestep_video_input
                        ),
                        "audio_prompt_timestep": (
                            prompt_timestep_audio
                            if prompt_timestep_audio_input is None
                            else prompt_timestep_audio_input
                        ),
                        "video_self_attention_mask": (
                            video_self_attention_mask
                            if video_self_attention_mask_input is None
                            else video_self_attention_mask_input
                        ),
                        "audio_self_attention_mask": (
                            audio_self_attention_mask
                            if audio_self_attention_mask_input is None
                            else audio_self_attention_mask_input
                        ),
                        "a2v_cross_attention_mask": (
                            a2v_cross_attention_mask
                            if a2v_cross_attention_mask_input is None
                            else a2v_cross_attention_mask_input
                        ),
                        "v2a_cross_attention_mask": (
                            v2a_cross_attention_mask
                            if v2a_cross_attention_mask_input is None
                            else v2a_cross_attention_mask_input
                        ),
                        "audio_replicated_for_sp": ctx.replicate_audio_for_sp,
                        "legacy_ltx23_one_stage_semantics": False,
                    }
                )
            if skip_video_self_attn_blocks is not None:
                kwargs["skip_video_self_attn_blocks"] = skip_video_self_attn_blocks
            if skip_audio_self_attn_blocks is not None:
                kwargs["skip_audio_self_attn_blocks"] = skip_audio_self_attn_blocks
            if disable_a2v_cross_attn:
                kwargs["disable_a2v_cross_attn"] = True
            if disable_v2a_cross_attn:
                kwargs["disable_v2a_cross_attn"] = True
            return kwargs

        # 5. Run the branch-specific LTX forward path and apply CFG/guider logic.
        prompt_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            ),
        )
        def build_stage1_step_inputs(
            video_latents: torch.Tensor,
            audio_latents: torch.Tensor,
            sigma_value: torch.Tensor,
        ) -> dict[str, object]:
            local_batch_size = int(video_latents.shape[0])
            video_hidden_states = video_latents.to(ctx.target_dtype)
            audio_hidden_states = audio_latents.to(ctx.target_dtype)
            if audio_hidden_states.ndim == 3:
                local_audio_num_frames = int(audio_hidden_states.shape[1])
            elif audio_hidden_states.ndim == 4:
                local_audio_num_frames = int(audio_hidden_states.shape[2])
            else:
                raise ValueError(
                    "Unexpected audio latents rank: "
                    f"{audio_hidden_states.ndim}, shape={tuple(audio_hidden_states.shape)}"
                )

            use_raw_sigma_timestep = (
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            )
            timestep_base = sigma_value.to(
                device=video_hidden_states.device, dtype=torch.float32
            )
            if not use_raw_sigma_timestep:
                timestep_base = interpolate_timestep_from_sigma(
                    sigma_value, video_hidden_states.device
                )

            timestep_local = timestep_base.expand(local_batch_size)
            if ctx.denoise_mask is not None:
                if use_raw_sigma_timestep:
                    timestep_video_local = timestep_local.view(
                        local_batch_size, *([1] * (ctx.denoise_mask.ndim - 1))
                    ) * ctx.denoise_mask
                else:
                    timestep_video_local = (
                        timestep_local.unsqueeze(-1) * ctx.denoise_mask.squeeze(-1)
                    )
            elif use_raw_sigma_timestep:
                timestep_video_local = timestep_local.view(
                    local_batch_size, 1, 1
                ).expand(local_batch_size, int(video_hidden_states.shape[1]), 1)
            else:
                timestep_video_local = timestep_local

            if use_raw_sigma_timestep and audio_hidden_states.ndim == 3:
                timestep_audio_local = timestep_local.view(
                    local_batch_size, 1, 1
                ).expand(local_batch_size, int(audio_hidden_states.shape[1]), 1)
            else:
                timestep_audio_local = timestep_local

            prompt_timestep_video_local = sigma_value.to(
                device=video_hidden_states.device, dtype=torch.float32
            ).expand(local_batch_size)
            prompt_timestep_audio_local = sigma_value.to(
                device=audio_hidden_states.device, dtype=torch.float32
            ).expand(local_batch_size)

            return {
                "hidden_states": video_hidden_states,
                "audio_hidden_states": audio_hidden_states,
                "timestep_video": timestep_video_local,
                "timestep_audio": timestep_audio_local,
                "prompt_timestep_video": prompt_timestep_video_local,
                "prompt_timestep_audio": prompt_timestep_audio_local,
                "audio_num_frames": local_audio_num_frames,
            }

        use_official_cfg_path = stage1_guider_params is None
        if use_official_cfg_path:
            encoder_hidden_states = batch.prompt_embeds[0]
            audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
            encoder_attention_mask = prompt_attention_mask
            if batch.do_classifier_free_guidance:
                latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
                audio_latent_model_input = torch.cat(
                    [audio_latent_model_input] * 2, dim=0
                )
                encoder_hidden_states = torch.cat(
                    [batch.negative_prompt_embeds[0], encoder_hidden_states], dim=0
                )
                audio_encoder_hidden_states = torch.cat(
                    [
                        batch.negative_audio_prompt_embeds[0],
                        audio_encoder_hidden_states,
                    ],
                    dim=0,
                )
                if encoder_attention_mask is not None:
                    encoder_attention_mask = torch.cat(
                        [
                            self._get_ltx_prompt_attention_mask(
                                batch,
                                is_ltx23_variant=(
                                    ctx.is_ltx23_variant
                                    and not ctx.use_ltx23_legacy_one_stage
                                ),
                                negative=True,
                            ),
                            encoder_attention_mask,
                        ],
                        dim=0,
                    )
                cfg_batch_size = int(latent_model_input.shape[0])
                timestep_video = self._repeat_batch_dim(timestep_video, cfg_batch_size)
                timestep_audio = self._repeat_batch_dim(timestep_audio, cfg_batch_size)
                if prompt_timestep_video is not None:
                    prompt_timestep_video = self._repeat_batch_dim(
                        prompt_timestep_video, cfg_batch_size
                    )
                if prompt_timestep_audio is not None:
                    prompt_timestep_audio = self._repeat_batch_dim(
                        prompt_timestep_audio, cfg_batch_size
                    )
                if video_self_attention_mask is not None:
                    video_self_attention_mask = self._repeat_batch_dim(
                        video_self_attention_mask, cfg_batch_size
                    )
                if audio_self_attention_mask is not None:
                    audio_self_attention_mask = self._repeat_batch_dim(
                        audio_self_attention_mask, cfg_batch_size
                    )
                if a2v_cross_attention_mask is not None:
                    a2v_cross_attention_mask = self._repeat_batch_dim(
                        a2v_cross_attention_mask, cfg_batch_size
                    )
                if v2a_cross_attention_mask is not None:
                    v2a_cross_attention_mask = self._repeat_batch_dim(
                        v2a_cross_attention_mask, cfg_batch_size
                    )

            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                model_video, model_audio = step.current_model(
                    **build_model_kwargs(
                        encoder_hidden_states=encoder_hidden_states,
                        audio_encoder_hidden_states=audio_encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                )

            model_video = model_video.float()
            model_audio = model_audio.float()
            if batch.do_classifier_free_guidance:
                model_video_uncond, model_video_text = model_video.chunk(2)
                model_audio_uncond, model_audio_text = model_audio.chunk(2)
                model_video = model_video_uncond + (
                    batch.guidance_scale * (model_video_text - model_video_uncond)
                )
                model_audio = model_audio_uncond + (
                    batch.guidance_scale * (model_audio_text - model_audio_uncond)
                )

            if self.sampler_name == "res2s":
                def _stage2_midpoint_model_call(
                    video_latents: torch.Tensor,
                    audio_latents: torch.Tensor,
                    sigma_value: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    step_inputs = build_stage1_step_inputs(
                        video_latents=video_latents,
                        audio_latents=audio_latents,
                        sigma_value=sigma_value,
                    )
                    with set_forward_context(
                        current_timestep=step.step_index,
                        attn_metadata=step.attn_metadata,
                    ):
                        mid_v, mid_a = step.current_model(
                            **build_model_kwargs(
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                hidden_states=step_inputs["hidden_states"],
                                audio_hidden_states=step_inputs["audio_hidden_states"],
                                timestep_video_input=step_inputs["timestep_video"],
                                timestep_audio_input=step_inputs["timestep_audio"],
                                prompt_timestep_video_input=step_inputs[
                                    "prompt_timestep_video"
                                ],
                                prompt_timestep_audio_input=step_inputs[
                                    "prompt_timestep_audio"
                                ],
                                audio_num_frames_input=int(
                                    step_inputs["audio_num_frames"]
                                ),
                            )
                        )
                    return mid_v.float(), mid_a.float()

                ctx.latents, ctx.audio_latents = self._ltx2_stage2_res2s_step(
                    ctx=ctx,
                    batch=batch,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    model_video_velocity=model_video,
                    model_audio_velocity=model_audio,
                    midpoint_model_call=_stage2_midpoint_model_call,
                )
            else:
                ctx.latents = self.scheduler.step(
                    model_video, step.t_device, ctx.latents, return_dict=False
                )[0]
                ctx.audio_latents = ctx.audio_scheduler.step(
                    model_audio, step.t_device, ctx.audio_latents, return_dict=False
                )[0]
                if ctx.denoise_mask is not None and ctx.clean_latent is not None:
                    ctx.latents = (
                        ctx.latents.float() * ctx.denoise_mask
                        + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
                    ).to(dtype=ctx.latents.dtype)
            self._maybe_dump_phase_step0_probe(
                batch,
                phase=phase,
                step_index=step.step_index,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                latent_model_input=latent_model_input,
                audio_latent_model_input=audio_latent_model_input,
                timestep_video=timestep_video,
                timestep_audio=timestep_audio,
                prompt_timestep_video=prompt_timestep_video,
                prompt_timestep_audio=prompt_timestep_audio,
                model_video=model_video,
                model_audio=model_audio,
                latents_after_step=ctx.latents,
                audio_latents_after_step=ctx.audio_latents,
            )
            ctx.latents = self.post_forward_for_ti2v_task(
                batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
            )
            return

        encoder_hidden_states = batch.prompt_embeds[0]
        audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
        encoder_attention_mask = prompt_attention_mask
        negative_encoder_hidden_states = batch.negative_prompt_embeds[0]
        negative_audio_encoder_hidden_states = batch.negative_audio_prompt_embeds[0]
        negative_encoder_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            ),
            negative=True,
        )

        video_skip = self._ltx2_should_skip_step(
            step.step_index, int(stage1_guider_params["video_skip_step"])
        )
        audio_skip = self._ltx2_should_skip_step(
            step.step_index, int(stage1_guider_params["audio_skip_step"])
        )
        need_perturbed = (
            float(stage1_guider_params["video_stg_scale"]) != 0.0
            or float(stage1_guider_params["audio_stg_scale"]) != 0.0
        )
        need_modality = (
            float(stage1_guider_params["video_modality_scale"]) != 1.0
            or float(stage1_guider_params["audio_modality_scale"]) != 1.0
        )

        def interpolate_timestep_from_sigma(
            sigma_value: torch.Tensor, device: torch.device
        ) -> torch.Tensor:
            sigma_value = sigma_value.to(device=device, dtype=torch.float32)
            scheduler_timesteps = getattr(self.scheduler, "timesteps", None)
            if (
                isinstance(scheduler_timesteps, torch.Tensor)
                and scheduler_timesteps.numel() > step.step_index + 1
            ):
                t_cur = scheduler_timesteps[step.step_index].to(
                    device=device, dtype=torch.float32
                )
                t_next = scheduler_timesteps[step.step_index + 1].to(
                    device=device, dtype=torch.float32
                )
                sigma_cur = sigma.to(device=device, dtype=torch.float32)
                sigma_next_local = sigma_next.to(device=device, dtype=torch.float32)
                denom = sigma_next_local - sigma_cur
                if bool(torch.abs(denom) < 1e-12):
                    return t_cur
                ratio = (sigma_value - sigma_cur) / denom
                return t_cur + ratio * (t_next - t_cur)
            sigma_cur = sigma.to(device=device, dtype=torch.float32)
            if bool(sigma_cur == 0):
                return torch.zeros((), device=device, dtype=torch.float32)
            return step.t_device.to(device=device, dtype=torch.float32) * (
                sigma_value / sigma_cur
            )

        def evaluate_stage1_guided_x0(
            *,
            video_latents: torch.Tensor,
            audio_latents: torch.Tensor,
            sigma_value: torch.Tensor,
            update_skip_cache: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            step_inputs = build_stage1_step_inputs(
                video_latents=video_latents,
                audio_latents=audio_latents,
                sigma_value=sigma_value,
            )
            local_batch_size = int(video_latents.shape[0])

            def cat_or_none(items: list[torch.Tensor | None]) -> torch.Tensor | None:
                if items[0] is None:
                    return None
                return torch.cat(items, dim=0)

            if ctx.use_ltx23_legacy_one_stage:
                with set_forward_context(
                    current_timestep=step.step_index, attn_metadata=step.attn_metadata
                ):
                    v_pos, a_v_pos = step.current_model(
                        **build_model_kwargs(
                            encoder_hidden_states=encoder_hidden_states,
                            audio_encoder_hidden_states=audio_encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            hidden_states=step_inputs["hidden_states"],
                            audio_hidden_states=step_inputs["audio_hidden_states"],
                            timestep_video_input=step_inputs["timestep_video"],
                            timestep_audio_input=step_inputs["timestep_audio"],
                            prompt_timestep_video_input=step_inputs[
                                "prompt_timestep_video"
                            ],
                            prompt_timestep_audio_input=step_inputs[
                                "prompt_timestep_audio"
                            ],
                            audio_num_frames_input=int(step_inputs["audio_num_frames"]),
                        )
                    )
                    v_neg, a_v_neg = step.current_model(
                        **build_model_kwargs(
                            encoder_hidden_states=negative_encoder_hidden_states,
                            audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
                            encoder_attention_mask=negative_encoder_attention_mask,
                            hidden_states=step_inputs["hidden_states"],
                            audio_hidden_states=step_inputs["audio_hidden_states"],
                            timestep_video_input=step_inputs["timestep_video"],
                            timestep_audio_input=step_inputs["timestep_audio"],
                            prompt_timestep_video_input=step_inputs[
                                "prompt_timestep_video"
                            ],
                            prompt_timestep_audio_input=step_inputs[
                                "prompt_timestep_audio"
                            ],
                            audio_num_frames_input=int(step_inputs["audio_num_frames"]),
                        )
                    )

                v_pos = v_pos.float()
                a_v_pos = a_v_pos.float()
                v_neg = v_neg.float()
                a_v_neg = a_v_neg.float()

                v_ptb = None
                a_v_ptb = None
                if need_perturbed:
                    with set_forward_context(
                        current_timestep=step.step_index,
                        attn_metadata=step.attn_metadata,
                    ):
                        v_ptb, a_v_ptb = step.current_model(
                            **build_model_kwargs(
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                hidden_states=step_inputs["hidden_states"],
                                audio_hidden_states=step_inputs["audio_hidden_states"],
                                timestep_video_input=step_inputs["timestep_video"],
                                timestep_audio_input=step_inputs["timestep_audio"],
                                prompt_timestep_video_input=step_inputs[
                                    "prompt_timestep_video"
                                ],
                                prompt_timestep_audio_input=step_inputs[
                                    "prompt_timestep_audio"
                                ],
                                audio_num_frames_input=int(
                                    step_inputs["audio_num_frames"]
                                ),
                                skip_video_self_attn_blocks=tuple(
                                    stage1_guider_params["video_stg_blocks"]
                                ),
                                skip_audio_self_attn_blocks=tuple(
                                    stage1_guider_params["audio_stg_blocks"]
                                ),
                            )
                        )
                    v_ptb = v_ptb.float()
                    a_v_ptb = a_v_ptb.float()

                v_mod = None
                a_v_mod = None
                if need_modality:
                    with set_forward_context(
                        current_timestep=step.step_index,
                        attn_metadata=step.attn_metadata,
                    ):
                        v_mod, a_v_mod = step.current_model(
                            **build_model_kwargs(
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                hidden_states=step_inputs["hidden_states"],
                                audio_hidden_states=step_inputs["audio_hidden_states"],
                                timestep_video_input=step_inputs["timestep_video"],
                                timestep_audio_input=step_inputs["timestep_audio"],
                                prompt_timestep_video_input=step_inputs[
                                    "prompt_timestep_video"
                                ],
                                prompt_timestep_audio_input=step_inputs[
                                    "prompt_timestep_audio"
                                ],
                                audio_num_frames_input=int(
                                    step_inputs["audio_num_frames"]
                                ),
                                disable_a2v_cross_attn=True,
                                disable_v2a_cross_attn=True,
                            )
                        )
                    v_mod = v_mod.float()
                    a_v_mod = a_v_mod.float()
            else:
                use_cfg_pair_batch = (
                    self._should_use_ltx23_two_stage_cfg_pair_batch(
                        batch=batch,
                        ctx=ctx,
                        server_args=server_args,
                    )
                    and not need_perturbed
                    and not need_modality
                )
                use_split_two_stage_guided_passes = self._should_use_split_two_stage_guided_passes(
                    batch=batch,
                    server_args=server_args,
                )

                if use_cfg_pair_batch:
                    expanded_batch_size = local_batch_size * 2
                    with set_forward_context(
                        current_timestep=step.step_index,
                        attn_metadata=step.attn_metadata,
                    ):
                        batched_video, batched_audio = step.current_model(
                            **build_model_kwargs(
                                encoder_hidden_states=torch.cat(
                                    [
                                        negative_encoder_hidden_states,
                                        encoder_hidden_states,
                                    ],
                                    dim=0,
                                ),
                                audio_encoder_hidden_states=torch.cat(
                                    [
                                        negative_audio_encoder_hidden_states,
                                        audio_encoder_hidden_states,
                                    ],
                                    dim=0,
                                ),
                                encoder_attention_mask=cat_or_none(
                                    [
                                        negative_encoder_attention_mask,
                                        encoder_attention_mask,
                                    ]
                                ),
                                hidden_states=self._repeat_batch_dim(
                                    step_inputs["hidden_states"], expanded_batch_size
                                ),
                                audio_hidden_states=self._repeat_batch_dim(
                                    step_inputs["audio_hidden_states"],
                                    expanded_batch_size,
                                ),
                                timestep_video_input=self._repeat_batch_dim(
                                    step_inputs["timestep_video"], expanded_batch_size
                                ),
                                timestep_audio_input=self._repeat_batch_dim(
                                    step_inputs["timestep_audio"], expanded_batch_size
                                ),
                                prompt_timestep_video_input=(
                                    None
                                    if step_inputs["prompt_timestep_video"] is None
                                    else self._repeat_batch_dim(
                                        step_inputs["prompt_timestep_video"],
                                        expanded_batch_size,
                                    )
                                ),
                                prompt_timestep_audio_input=(
                                    None
                                    if step_inputs["prompt_timestep_audio"] is None
                                    else self._repeat_batch_dim(
                                        step_inputs["prompt_timestep_audio"],
                                        expanded_batch_size,
                                    )
                                ),
                                audio_num_frames_input=int(
                                    step_inputs["audio_num_frames"]
                                ),
                                video_coords_input=(
                                    None
                                    if video_coords is None
                                    else self._repeat_batch_dim(
                                        video_coords, expanded_batch_size
                                    )
                                ),
                                audio_coords_input=(
                                    None
                                    if audio_coords is None
                                    else self._repeat_batch_dim(
                                        audio_coords, expanded_batch_size
                                    )
                                ),
                                video_self_attention_mask_input=(
                                    None
                                    if video_self_attention_mask is None
                                    else self._repeat_batch_dim(
                                        video_self_attention_mask,
                                        expanded_batch_size,
                                    )
                                ),
                                audio_self_attention_mask_input=(
                                    None
                                    if audio_self_attention_mask is None
                                    else self._repeat_batch_dim(
                                        audio_self_attention_mask,
                                        expanded_batch_size,
                                    )
                                ),
                                a2v_cross_attention_mask_input=(
                                    None
                                    if a2v_cross_attention_mask is None
                                    else self._repeat_batch_dim(
                                        a2v_cross_attention_mask,
                                        expanded_batch_size,
                                    )
                                ),
                                v2a_cross_attention_mask_input=(
                                    None
                                    if v2a_cross_attention_mask is None
                                    else self._repeat_batch_dim(
                                        v2a_cross_attention_mask,
                                        expanded_batch_size,
                                    )
                                ),
                            )
                        )
                    batched_video = batched_video.float()
                    batched_audio = batched_audio.float()
                    v_neg, v_pos = batched_video.chunk(2, dim=0)
                    a_v_neg, a_v_pos = batched_audio.chunk(2, dim=0)
                    v_ptb = None
                    a_v_ptb = None
                    v_mod = None
                    a_v_mod = None
                else:
                    pass_specs: list[
                        tuple[
                            str,
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor | None,
                            dict[str, object],
                        ]
                    ] = [
                        (
                            "cond",
                            encoder_hidden_states,
                            audio_encoder_hidden_states,
                            encoder_attention_mask,
                            {
                                "skip_video_self_attn_blocks": (),
                                "skip_audio_self_attn_blocks": (),
                                "skip_a2v_cross_attn": False,
                                "skip_v2a_cross_attn": False,
                            },
                        ),
                        (
                            "neg",
                            negative_encoder_hidden_states,
                            negative_audio_encoder_hidden_states,
                            negative_encoder_attention_mask,
                            {
                                "skip_video_self_attn_blocks": (),
                                "skip_audio_self_attn_blocks": (),
                                "skip_a2v_cross_attn": False,
                                "skip_v2a_cross_attn": False,
                            },
                        ),
                    ]
                    if need_perturbed:
                        pass_specs.append(
                            (
                                "perturbed",
                                encoder_hidden_states,
                                audio_encoder_hidden_states,
                                encoder_attention_mask,
                                {
                                    "skip_video_self_attn_blocks": tuple(
                                        stage1_guider_params["video_stg_blocks"]
                                    ),
                                    "skip_audio_self_attn_blocks": tuple(
                                        stage1_guider_params["audio_stg_blocks"]
                                    ),
                                    "skip_a2v_cross_attn": False,
                                    "skip_v2a_cross_attn": False,
                                },
                            )
                        )
                    if need_modality:
                        pass_specs.append(
                            (
                                "modality",
                                encoder_hidden_states,
                                audio_encoder_hidden_states,
                                encoder_attention_mask,
                                {
                                    "skip_video_self_attn_blocks": (),
                                    "skip_audio_self_attn_blocks": (),
                                    "skip_a2v_cross_attn": True,
                                    "skip_v2a_cross_attn": True,
                                },
                            )
                        )

                    num_passes = len(pass_specs)
                    expanded_batch_size = local_batch_size * num_passes
                    perturbation_configs = tuple(
                        perturbation_config
                        for _, _, _, _, perturbation_config in pass_specs
                        for _ in range(local_batch_size)
                    )
                    batched_hidden_states = self._repeat_batch_dim(
                        step_inputs["hidden_states"], expanded_batch_size
                    )
                    batched_audio_hidden_states = self._repeat_batch_dim(
                        step_inputs["audio_hidden_states"], expanded_batch_size
                    )
                    batched_encoder_hidden_states = torch.cat(
                        [item[1] for item in pass_specs], dim=0
                    )
                    batched_audio_encoder_hidden_states = torch.cat(
                        [item[2] for item in pass_specs], dim=0
                    )
                    batched_timestep_video = self._repeat_batch_dim(
                        step_inputs["timestep_video"], expanded_batch_size
                    )
                    batched_timestep_audio = self._repeat_batch_dim(
                        step_inputs["timestep_audio"], expanded_batch_size
                    )
                    batched_prompt_timestep_video = (
                        None
                        if step_inputs["prompt_timestep_video"] is None
                        else self._repeat_batch_dim(
                            step_inputs["prompt_timestep_video"], expanded_batch_size
                        )
                    )
                    batched_prompt_timestep_audio = (
                        None
                        if step_inputs["prompt_timestep_audio"] is None
                        else self._repeat_batch_dim(
                            step_inputs["prompt_timestep_audio"], expanded_batch_size
                        )
                    )
                    batched_encoder_attention_mask = cat_or_none(
                        [item[3] for item in pass_specs]
                    )
                    batched_audio_encoder_attention_mask = cat_or_none(
                        [item[3] for item in pass_specs]
                    )
                    batched_video_coords = (
                        None
                        if video_coords is None
                        else self._repeat_batch_dim(video_coords, expanded_batch_size)
                    )
                    batched_audio_coords = (
                        None
                        if audio_coords is None
                        else self._repeat_batch_dim(audio_coords, expanded_batch_size)
                    )
                    batched_video_self_attention_mask = (
                        None
                        if video_self_attention_mask is None
                        else self._repeat_batch_dim(
                            video_self_attention_mask, expanded_batch_size
                        )
                    )
                    batched_audio_self_attention_mask = (
                        None
                        if audio_self_attention_mask is None
                        else self._repeat_batch_dim(
                            audio_self_attention_mask, expanded_batch_size
                        )
                    )
                    batched_a2v_cross_attention_mask = (
                        None
                        if a2v_cross_attention_mask is None
                        else self._repeat_batch_dim(
                            a2v_cross_attention_mask, expanded_batch_size
                        )
                    )
                    batched_v2a_cross_attention_mask = (
                        None
                        if v2a_cross_attention_mask is None
                        else self._repeat_batch_dim(
                            v2a_cross_attention_mask, expanded_batch_size
                        )
                    )
                    if use_split_two_stage_guided_passes:
                        split_sizes = [1] * expanded_batch_size

                        def split_or_none(
                            tensor: torch.Tensor | None,
                        ) -> list[torch.Tensor | None]:
                            if tensor is None:
                                return [None] * len(split_sizes)
                            return list(tensor.split(split_sizes, dim=0))

                        batched_video_chunks = []
                        batched_audio_chunks = []
                        with set_forward_context(
                            current_timestep=step.step_index,
                            attn_metadata=step.attn_metadata,
                        ):
                            for (
                                hidden_states_chunk,
                                audio_hidden_states_chunk,
                                encoder_hidden_states_chunk,
                                audio_encoder_hidden_states_chunk,
                                timestep_video_chunk,
                                timestep_audio_chunk,
                                prompt_timestep_video_chunk,
                                prompt_timestep_audio_chunk,
                                encoder_attention_mask_chunk,
                                audio_encoder_attention_mask_chunk,
                                video_coords_chunk,
                                audio_coords_chunk,
                                video_self_attention_mask_chunk,
                                audio_self_attention_mask_chunk,
                                a2v_cross_attention_mask_chunk,
                                v2a_cross_attention_mask_chunk,
                                perturbation_config_chunk,
                            ) in zip(
                                batched_hidden_states.split(split_sizes, dim=0),
                                batched_audio_hidden_states.split(split_sizes, dim=0),
                                batched_encoder_hidden_states.split(split_sizes, dim=0),
                                batched_audio_encoder_hidden_states.split(
                                    split_sizes, dim=0
                                ),
                                batched_timestep_video.split(split_sizes, dim=0),
                                batched_timestep_audio.split(split_sizes, dim=0),
                                split_or_none(batched_prompt_timestep_video),
                                split_or_none(batched_prompt_timestep_audio),
                                split_or_none(batched_encoder_attention_mask),
                                split_or_none(batched_audio_encoder_attention_mask),
                                split_or_none(batched_video_coords),
                                split_or_none(batched_audio_coords),
                                split_or_none(batched_video_self_attention_mask),
                                split_or_none(batched_audio_self_attention_mask),
                                split_or_none(batched_a2v_cross_attention_mask),
                                split_or_none(batched_v2a_cross_attention_mask),
                                ((cfg,) for cfg in perturbation_configs),
                                strict=True,
                            ):
                                video_chunk, audio_chunk = step.current_model(
                                    hidden_states=hidden_states_chunk,
                                    audio_hidden_states=audio_hidden_states_chunk,
                                    encoder_hidden_states=encoder_hidden_states_chunk,
                                    audio_encoder_hidden_states=audio_encoder_hidden_states_chunk,
                                    timestep=timestep_video_chunk,
                                    audio_timestep=timestep_audio_chunk,
                                    prompt_timestep=prompt_timestep_video_chunk,
                                    audio_prompt_timestep=prompt_timestep_audio_chunk,
                                    encoder_attention_mask=encoder_attention_mask_chunk,
                                    audio_encoder_attention_mask=audio_encoder_attention_mask_chunk,
                                    num_frames=ctx.latent_num_frames_for_model,
                                    height=ctx.latent_height,
                                    width=ctx.latent_width,
                                    fps=batch.fps,
                                    audio_num_frames=int(step_inputs["audio_num_frames"]),
                                    video_coords=video_coords_chunk,
                                    audio_coords=audio_coords_chunk,
                                video_self_attention_mask=video_self_attention_mask_chunk,
                                audio_self_attention_mask=audio_self_attention_mask_chunk,
                                a2v_cross_attention_mask=a2v_cross_attention_mask_chunk,
                                v2a_cross_attention_mask=v2a_cross_attention_mask_chunk,
                                audio_replicated_for_sp=ctx.replicate_audio_for_sp,
                                _sglang_internal_probe_path=internal_probe_path,
                                _sglang_internal_probe_phase=phase,
                                perturbation_configs=perturbation_config_chunk,
                                return_latents=False,
                                return_dict=False,
                            )
                                batched_video_chunks.append(video_chunk)
                                batched_audio_chunks.append(audio_chunk)

                        batched_video = torch.cat(batched_video_chunks, dim=0)
                        batched_audio = torch.cat(batched_audio_chunks, dim=0)
                    else:
                        with set_forward_context(
                            current_timestep=step.step_index,
                            attn_metadata=step.attn_metadata,
                        ):
                            batched_video, batched_audio = step.current_model(
                                hidden_states=batched_hidden_states,
                                audio_hidden_states=batched_audio_hidden_states,
                                encoder_hidden_states=batched_encoder_hidden_states,
                                audio_encoder_hidden_states=batched_audio_encoder_hidden_states,
                                timestep=batched_timestep_video,
                                audio_timestep=batched_timestep_audio,
                                prompt_timestep=batched_prompt_timestep_video,
                                audio_prompt_timestep=batched_prompt_timestep_audio,
                                encoder_attention_mask=batched_encoder_attention_mask,
                                audio_encoder_attention_mask=batched_audio_encoder_attention_mask,
                                num_frames=ctx.latent_num_frames_for_model,
                                height=ctx.latent_height,
                                width=ctx.latent_width,
                                fps=batch.fps,
                                audio_num_frames=int(step_inputs["audio_num_frames"]),
                                video_coords=batched_video_coords,
                                audio_coords=batched_audio_coords,
                                video_self_attention_mask=batched_video_self_attention_mask,
                                audio_self_attention_mask=batched_audio_self_attention_mask,
                                a2v_cross_attention_mask=batched_a2v_cross_attention_mask,
                                v2a_cross_attention_mask=batched_v2a_cross_attention_mask,
                                audio_replicated_for_sp=ctx.replicate_audio_for_sp,
                                _sglang_internal_probe_path=internal_probe_path,
                                _sglang_internal_probe_phase=phase,
                                perturbation_configs=perturbation_configs,
                                return_latents=False,
                                return_dict=False,
                            )

                    batched_video = batched_video.float()
                    batched_audio = batched_audio.float()
                    sigma_batch = sigma_value.to(
                        device=batched_hidden_states.device, dtype=torch.float32
                    ).expand(expanded_batch_size)
                    self._maybe_dump_phase_transformer_step0_probe(
                        batch,
                        phase=phase,
                        step_index=step.step_index,
                        payload={
                            "pass_names": [item[0] for item in pass_specs],
                            "video": {
                                "latent": batched_hidden_states,
                                "sigma": sigma_batch,
                                "timesteps": batched_timestep_video,
                                "positions": batched_video_coords,
                                "context": batched_encoder_hidden_states,
                                "enabled": True,
                                "context_mask": batched_encoder_attention_mask,
                                "attention_mask": batched_video_self_attention_mask,
                            },
                            "audio": {
                                "latent": batched_audio_hidden_states,
                                "sigma": sigma_batch.to(
                                    device=batched_audio_hidden_states.device
                                ),
                                "timesteps": batched_timestep_audio,
                                "positions": batched_audio_coords,
                                "context": batched_audio_encoder_hidden_states,
                                "enabled": True,
                                "context_mask": batched_audio_encoder_attention_mask,
                                "attention_mask": batched_audio_self_attention_mask,
                            },
                            "prompt_timestep_video": batched_prompt_timestep_video,
                            "prompt_timestep_audio": batched_prompt_timestep_audio,
                            "a2v_cross_attention_mask": batched_a2v_cross_attention_mask,
                            "v2a_cross_attention_mask": batched_v2a_cross_attention_mask,
                            "perturbation_configs": perturbation_configs,
                            "model_video": batched_video,
                            "model_audio": batched_audio,
                        },
                    )
                    pass_outputs = {
                        pass_name: (
                            video_chunk,
                            audio_chunk,
                        )
                        for (pass_name, _, _, _, _), video_chunk, audio_chunk in zip(
                            pass_specs,
                            batched_video.chunk(num_passes, dim=0),
                            batched_audio.chunk(num_passes, dim=0),
                            strict=True,
                        )
                    }
                    v_pos, a_v_pos = pass_outputs["cond"]
                    v_neg, a_v_neg = pass_outputs["neg"]
                    v_ptb, a_v_ptb = pass_outputs.get("perturbed", (None, None))
                    v_mod, a_v_mod = pass_outputs.get("modality", (None, None))

            sigma_value_float = float(sigma_value.item())
            video_sigma_for_x0: float | torch.Tensor = sigma_value_float
            if ctx.denoise_mask is not None:
                video_sigma_for_x0 = sigma_value.to(
                    device=video_latents.device, dtype=torch.float32
                ) * ctx.denoise_mask.squeeze(-1)

            denoised_video_local = self._ltx2_velocity_to_x0(
                video_latents, v_pos, video_sigma_for_x0
            )
            denoised_audio_local = self._ltx2_velocity_to_x0(
                audio_latents, a_v_pos, sigma_value_float
            )
            denoised_video_neg = self._ltx2_velocity_to_x0(
                video_latents, v_neg, video_sigma_for_x0
            )
            denoised_audio_neg = self._ltx2_velocity_to_x0(
                audio_latents, a_v_neg, sigma_value_float
            )
            denoised_video_perturbed = (
                None
                if v_ptb is None
                else self._ltx2_velocity_to_x0(
                    video_latents, v_ptb, video_sigma_for_x0
                )
            )
            denoised_audio_perturbed = (
                None
                if a_v_ptb is None
                else self._ltx2_velocity_to_x0(
                    audio_latents, a_v_ptb, sigma_value_float
                )
            )
            denoised_video_modality = (
                None
                if v_mod is None
                else self._ltx2_velocity_to_x0(
                    video_latents, v_mod, video_sigma_for_x0
                )
            )
            denoised_audio_modality = (
                None
                if a_v_mod is None
                else self._ltx2_velocity_to_x0(
                    audio_latents, a_v_mod, sigma_value_float
                )
            )

            guided_video = self._ltx2_calculate_guided_x0(
                cond=denoised_video_local,
                uncond_text=denoised_video_neg,
                uncond_perturbed=(
                    denoised_video_perturbed
                    if denoised_video_perturbed is not None
                    else 0.0
                ),
                uncond_modality=(
                    denoised_video_modality
                    if denoised_video_modality is not None
                    else 0.0
                ),
                cfg_scale=float(stage1_guider_params["video_cfg_scale"]),
                stg_scale=float(stage1_guider_params["video_stg_scale"]),
                rescale_scale=float(stage1_guider_params["video_rescale_scale"]),
                modality_scale=float(stage1_guider_params["video_modality_scale"]),
            )
            if video_skip and ctx.last_denoised_video is not None:
                denoised_video_local = ctx.last_denoised_video
            else:
                denoised_video_local = guided_video
                if update_skip_cache:
                    ctx.last_denoised_video = guided_video

            guided_audio = self._ltx2_calculate_guided_x0(
                cond=denoised_audio_local,
                uncond_text=denoised_audio_neg,
                uncond_perturbed=(
                    denoised_audio_perturbed
                    if denoised_audio_perturbed is not None
                    else 0.0
                ),
                uncond_modality=(
                    denoised_audio_modality
                    if denoised_audio_modality is not None
                    else 0.0
                ),
                cfg_scale=float(stage1_guider_params["audio_cfg_scale"]),
                stg_scale=float(stage1_guider_params["audio_stg_scale"]),
                rescale_scale=float(stage1_guider_params["audio_rescale_scale"]),
                modality_scale=float(stage1_guider_params["audio_modality_scale"]),
            )
            if audio_skip and ctx.last_denoised_audio is not None:
                denoised_audio_local = ctx.last_denoised_audio
            else:
                denoised_audio_local = guided_audio
                if update_skip_cache:
                    ctx.last_denoised_audio = guided_audio

            denoised_video_local = self._ltx2_apply_clean_latent_mask(
                denoised_video_local, ctx
            )
            return denoised_video_local, denoised_audio_local

        sigma_val = float(sigma.item())
        denoised_video, denoised_audio = evaluate_stage1_guided_x0(
            video_latents=ctx.latents,
            audio_latents=ctx.audio_latents,
            sigma_value=sigma,
            update_skip_cache=True,
        )

        if self.sampler_name == "res2s":
            if sigma_val == 0.0 or float(sigma_next.item()) == 0.0:
                next_video_latents = denoised_video.to(dtype=ctx.latents.dtype)
                next_audio_latents = denoised_audio.to(dtype=ctx.audio_latents.dtype)
            else:
                # Match official res2s: compute h / coefficients in fp64
                sigma_d = sigma.double()
                sigma_next_d = sigma_next.double()
                h = -torch.log(torch.clamp(sigma_next_d / sigma_d, min=1e-12))
                a21, b1, b2 = self._ltx2_get_res2s_coefficients(h)
                sub_sigma = torch.sqrt(torch.clamp(sigma_d * sigma_next_d, min=0.0))

                anchor_video = ctx.latents.double()
                anchor_audio = ctx.audio_latents.double()
                eps1_video = denoised_video.double() - anchor_video
                eps1_audio = denoised_audio.double() - anchor_audio

                midpoint_video_deterministic = anchor_video + h * a21 * eps1_video
                midpoint_audio_deterministic = anchor_audio + h * a21 * eps1_audio

                midpoint_video_latents = self._ltx2_res2s_sde_step(
                    sample=anchor_video,
                    denoised_sample=midpoint_video_deterministic,
                    sigma=sigma_d,
                    sigma_next=sub_sigma,
                    noise=(
                        self._ltx2_res2s_noise_like(
                            ctx.latents, ctx, substep=True
                        ).float()
                        if ctx.use_native_hq_res2s_sde_noise
                        else self._randn_like_with_batch_generators(
                            ctx.latents, batch
                        ).float()
                    ),
                )
                midpoint_audio_latents = self._ltx2_res2s_sde_step(
                    sample=anchor_audio,
                    denoised_sample=midpoint_audio_deterministic,
                    sigma=sigma_d,
                    sigma_next=sub_sigma,
                    noise=(
                        self._ltx2_res2s_noise_like(
                            ctx.audio_latents, ctx, substep=True
                        ).float()
                        if ctx.use_native_hq_res2s_sde_noise
                        else self._randn_like_with_batch_generators(
                            ctx.audio_latents, batch
                        ).float()
                    ),
                )

                midpoint_video_latents = self._ltx2_apply_clean_latent_mask(
                    midpoint_video_latents.to(dtype=ctx.latents.dtype),
                    ctx,
                )
                midpoint_audio_latents = midpoint_audio_latents.to(
                    dtype=ctx.audio_latents.dtype
                )

                # Match official bongmath: iterative anchor refinement when h
                # is small and sigma is far from 0. Adjusts anchor to reduce
                # RK2 truncation error. Active on first distilled step only.
                if float(h.item()) < 0.5 and sigma_val > 0.03:
                    x_mid_v = midpoint_video_latents.double()
                    x_mid_a = midpoint_audio_latents.double()
                    for _ in range(100):
                        anchor_video = x_mid_v - h * a21 * eps1_video
                        eps1_video = denoised_video.double() - anchor_video
                        anchor_audio = x_mid_a - h * a21 * eps1_audio
                        eps1_audio = denoised_audio.double() - anchor_audio

                midpoint_denoised_video, midpoint_denoised_audio = (
                    evaluate_stage1_guided_x0(
                        video_latents=midpoint_video_latents,
                        audio_latents=midpoint_audio_latents,
                        sigma_value=sub_sigma,
                        update_skip_cache=False,
                    )
                )
                eps2_video = midpoint_denoised_video.double() - anchor_video
                eps2_audio = midpoint_denoised_audio.double() - anchor_audio

                next_video_deterministic = anchor_video + h * (
                    b1 * eps1_video + b2 * eps2_video
                )
                next_audio_deterministic = anchor_audio + h * (
                    b1 * eps1_audio + b2 * eps2_audio
                )

                next_video_latents = self._ltx2_res2s_sde_step(
                    sample=anchor_video,
                    denoised_sample=next_video_deterministic,
                    sigma=sigma_d,
                    sigma_next=sigma_next_d,
                    noise=(
                        self._ltx2_res2s_noise_like(
                            ctx.latents, ctx, substep=False
                        ).float()
                        if ctx.use_native_hq_res2s_sde_noise
                        else self._randn_like_with_batch_generators(
                            ctx.latents, batch
                        ).float()
                    ),
                )
                next_audio_latents = self._ltx2_res2s_sde_step(
                    sample=anchor_audio,
                    denoised_sample=next_audio_deterministic,
                    sigma=sigma_d,
                    sigma_next=sigma_next_d,
                    noise=(
                        self._ltx2_res2s_noise_like(
                            ctx.audio_latents, ctx, substep=False
                        ).float()
                        if ctx.use_native_hq_res2s_sde_noise
                        else self._randn_like_with_batch_generators(
                            ctx.audio_latents, batch
                        ).float()
                    ),
                )

                next_video_latents = self._ltx2_apply_clean_latent_mask(
                    next_video_latents.to(dtype=ctx.latents.dtype),
                    ctx,
                )
                next_audio_latents = next_audio_latents.to(
                    dtype=ctx.audio_latents.dtype
                )
        else:
            if sigma_val == 0.0:
                v_video = torch.zeros_like(denoised_video)
                v_audio = torch.zeros_like(denoised_audio)
            else:
                v_video = (
                    (ctx.latents.float() - denoised_video.float()) / sigma_val
                ).to(ctx.latents.dtype)
                v_audio = (
                    (ctx.audio_latents.float() - denoised_audio.float()) / sigma_val
                ).to(ctx.audio_latents.dtype)

            next_video_latents = (ctx.latents.float() + v_video.float() * dt).to(
                dtype=ctx.latents.dtype
            )
            next_audio_latents = (
                ctx.audio_latents.float() + v_audio.float() * dt
            ).to(dtype=ctx.audio_latents.dtype)

        ctx.latents = next_video_latents
        ctx.audio_latents = next_audio_latents
        ctx.latents = self.post_forward_for_ti2v_task(
            batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
        )

    def _record_trajectory(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Record audio trajectory alongside the base video trajectory."""
        super()._record_trajectory(ctx, step, batch, server_args)
        if batch.return_trajectory_latents and ctx.audio_latents is not None:
            ctx.trajectory_audio_latents.append(ctx.audio_latents)

    def _finalize_denoising_loop(
        self, ctx: LTX2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Expose audio latents before delegating to AV-aware postprocessing."""
        batch.audio_latents = ctx.audio_latents
        self._post_denoising_loop(
            batch=batch,
            latents=ctx.latents,
            trajectory_latents=ctx.trajectory_latents,
            trajectory_timesteps=ctx.trajectory_timesteps,
            trajectory_audio_latents=ctx.trajectory_audio_latents,
            server_args=server_args,
            is_warmup=ctx.is_warmup,
        )

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        server_args: ServerArgs,
        trajectory_audio_latents: list | None = None,
        is_warmup: bool = False,
        *args,
        **kwargs,
    ):
        """Trim SP token padding before delegating to the base finalizer."""
        if trajectory_audio_latents:
            batch.trajectory_audio_latents = torch.stack(
                trajectory_audio_latents, dim=1
            ).cpu()
        latents = self._truncate_sp_padded_token_latents(batch, latents)
        super()._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            server_args=server_args,
            is_warmup=is_warmup,
        )

    def _get_prompt_embeds_validator(self, batch: Req):
        """Allow either tensor or list prompt embeddings for LTX-2 prompts."""
        del batch
        return lambda x: V.is_tensor(x) or V.list_not_empty(x)

    def _get_negative_prompt_embeds_validator(self, batch: Req):
        """Allow either tensor or list negative prompt embeddings for LTX-2 CFG."""
        return (
            lambda x: (not batch.do_classifier_free_guidance)
            or V.is_tensor(x)
            or V.list_not_empty(x)
        )
