# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def _freeze_batch_signature_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(k): _freeze_batch_signature_value(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_batch_signature_value(v) for v in value)
    return repr(value)


def _same_fields(base_req: "Req", ref_req: "Req", *names: str) -> bool:
    return all(
        getattr(base_req, name, None) == getattr(ref_req, name, None) for name in names
    )


def _same_model_specific_kwargs(base_req: "Req", ref_req: "Req") -> bool:
    return _freeze_batch_signature_value((base_req.extra or {}).get("diffusers_kwargs")) == _freeze_batch_signature_value(
        (ref_req.extra or {}).get("diffusers_kwargs")
    )


def can_batch_prompt_only_diffusion_requests(base_req: "Req", ref_req: "Req") -> bool:
    """
        A batching strategy for text-only diffusion requests.

        Prompts, negative prompts, request ids, output paths, output file names,
        and seeds may differ across the batched requests. Fields that affect the
        execution contract or merged output handling must remain identical.

    """
    if base_req.is_warmup or ref_req.is_warmup:
        return False

    if not isinstance(base_req.prompt, str) or not isinstance(ref_req.prompt, str):
        return False

    if base_req.image_path is not None or ref_req.image_path is not None:
        return False

    if base_req.return_file_paths_only != ref_req.return_file_paths_only:
        return False

    return (
        _same_fields(
            base_req,
            ref_req,
            "data_type",
            "audio_path",
            "pose_video_path",
            "num_clip",
            "generator_device",
            "num_frames",
            "num_frames_round_down",
            "height",
            "width",
            "fps",
            "num_inference_steps",
            "guidance_scale",
            "guidance_scale_2",
            "true_cfg_scale",
            "guidance_rescale",
            "cfg_normalization",
            "boundary_ratio",
            "enable_teacache",
            "enable_sequence_shard",
            "num_outputs_per_prompt",
            "save_output",
            "return_frames",
            "return_trajectory_latents",
            "return_trajectory_decoded",
            "return_file_paths_only",
            "output_quality",
            "output_compression",
            "enable_frame_interpolation",
            "frame_interpolation_exp",
            "frame_interpolation_scale",
            "frame_interpolation_model_path",
            "enable_upscaling",
            "upscaling_model_path",
            "upscaling_scale",
            "profile",
            "num_profiled_timesteps",
            "profile_all_stages",
            "debug",
            "perf_dump_path",
            "no_override_protected_fields",
            "adjust_frames",
            "suppress_logs",
        )
        and _freeze_batch_signature_value(getattr(base_req, "teacache_params", None))
        == _freeze_batch_signature_value(getattr(ref_req, "teacache_params", None))
        and _same_model_specific_kwargs(base_req, ref_req)
    )
