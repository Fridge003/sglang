from __future__ import annotations

import argparse
import dataclasses
from contextlib import contextmanager
from pathlib import Path

import torch

try:
    import ltx_pipelines.ti2vid_two_stages_hq as ltx2_hq_module
    import ltx_pipelines.utils.blocks as ltx2_blocks_module
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
    from ltx_core.text_encoders.gemma.encoders import encoder_configurator as gemma_encoder_configurator
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
    from ltx_pipelines.utils.media_io import encode_video
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3RotaryEmbedding,
        Gemma3TextScaledWordEmbedding,
    )
    from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings
except ImportError as exc:  # pragma: no cover - manual helper
    raise SystemExit(
        "This script must run against the official LTX-2 repo. "
        "Set PYTHONPATH to include packages/ltx-core/src and packages/ltx-pipelines/src."
    ) from exc


PR23366_SPONGEBOB_PROMPT = """00:00 - 00:02.5
Visual / Action:
SpongeBob stands in front of a mirror in his living room. He is wearing a brand new jellyfish-shaped hat with two glowing jellyfish eyes on top. He turns his head left and right, admiring himself with a big proud smile.

Dialogue:
SpongeBob (to himself, cheerful): "Perfect!"

SFX:
Upbeat whistling tune (light and bouncy), soft electric buzz from the hat's glowing eyes.

00:02.5 - 00:05
Visual / Action:
Suddenly, Patrick jumps out from behind SpongeBob, clapping his hands loudly and shouting. SpongeBob gets so scared that he jumps straight up, and the jellyfish hat flies off his head into the air.

Dialogue:
Patrick (loud and blunt): "Ugly hat!"

SFX:
Short dramatic sting - "DAH!" (comedy suspense), followed by a "whoosh" sound as the hat flies upward.

00:05 - 00:07.5
Visual / Action:
The hat falls down and lands perfectly on Patrick's head. Immediately, the hat's jellyfish eyes turn from yellow to red, and it makes an angry buzzing sound. Then, with a soft "poof", the hat sprays a small blob of pink goo directly onto Patrick's face.

Dialogue:
Patrick (confused, muffled): "Does it... hate me?"

SFX:
Warning buzz (low to high pitch), followed by a wet "plop" / "pfft" of the goo spraying, and a comedic slide whistle descending.

00:07.5 - 00:10
Visual / Action:
Patrick sticks out his tongue and licks the pink goo off his cheek. His eyes suddenly light up with joy, and he grins widely. SpongeBob doubles over, holding his stomach and laughing.

Dialogue:
Patrick (excited, mouth half-full): "It's strawberry flavored! Do it again!"

SFX:
Licking sound "lap", then a cheerful ukulele jingle (4 quick notes) + a final "ding!"
"""

WIDTH = 768
HEIGHT = 512
NUM_FRAMES = 121
FRAME_RATE = 25.0
NUM_INFERENCE_STEPS = 30
DISTILLED_LORA_STAGE_1 = 0.25
DISTILLED_LORA_STAGE_2 = 0.5

VIDEO_GUIDER_PARAMS = MultiModalGuiderParams(
    cfg_scale=3.0,
    stg_scale=0.0,
    rescale_scale=0.45,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=[],
)
AUDIO_GUIDER_PARAMS = MultiModalGuiderParams(
    cfg_scale=7.0,
    stg_scale=0.0,
    rescale_scale=1.0,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=[],
)

_STAGE2_BLOCK_PROBE_STATE: dict[str, object] = {
    "enabled": False,
    "probe_dir": None,
    "phase": "stage2",
    "num_blocks": 0,
    "selected_blocks": set(),
    "prompt_call_counts": {},
    "prompt_inputs": {},
    "prompt_deltas": {},
}


def _restore_missing_gemma_config_attrs(model: torch.nn.Module) -> None:
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    if text_config is None:
        return

    defaults = GEMMA3_CONFIG_FOR_LTX.text_config
    if not hasattr(text_config, "rope_local_base_freq"):
        text_config.rope_local_base_freq = defaults.rope_local_base_freq
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict) or "rope_type" not in rope_scaling:
        text_config.rope_scaling = dataclasses.asdict(defaults.rope_scaling)


def _materialize_known_meta_buffers(model: torch.nn.Module, device: torch.device) -> None:
    _restore_missing_gemma_config_attrs(model)
    module_map = dict(model.named_modules())

    for full_name, buffer in model.named_buffers():
        if str(buffer.device) != "meta":
            continue

        module_name, buffer_name = full_name.rsplit(".", 1)
        module = module_map[module_name]

        if isinstance(module, SiglipVisionEmbeddings) and buffer_name == "position_ids":
            module._buffers[buffer_name] = torch.arange(
                module.num_positions, device=device, dtype=torch.long
            ).expand((1, -1))
            continue

        if isinstance(module, Gemma3TextScaledWordEmbedding) and buffer_name == "embed_scale":
            module._buffers[buffer_name] = torch.tensor(
                module.embedding_dim**0.5, device=device
            )
            continue

        if isinstance(module, Gemma3RotaryEmbedding) and buffer_name == "inv_freq":
            inv_freq, attention_scaling = module.rope_init_fn(module.config, device)
            module._buffers[buffer_name] = inv_freq
            module.original_inv_freq = inv_freq
            module.attention_scaling = attention_scaling


def _patch_official_builder_for_gemma_buffers() -> None:
    if getattr(SingleGPUModelBuilder, "_pr23366_meta_patch_applied", False):
        return

    original_return_model = SingleGPUModelBuilder._return_model

    def _patched_return_model(self, meta_model: torch.nn.Module, device: torch.device):
        _materialize_known_meta_buffers(meta_model, device)
        return original_return_model(self, meta_model, device)

    original_create_and_populate = gemma_encoder_configurator.create_and_populate

    def _patched_create_and_populate(module):
        _restore_missing_gemma_config_attrs(module.model)
        return original_create_and_populate(module)

    gemma_encoder_configurator.create_and_populate = _patched_create_and_populate
    gemma_encoder_configurator.GEMMA_MODEL_OPS = (
        gemma_encoder_configurator.GEMMA_MODEL_OPS._replace(
            mutator=_patched_create_and_populate
        )
    )
    ltx2_blocks_module.GEMMA_MODEL_OPS = gemma_encoder_configurator.GEMMA_MODEL_OPS
    SingleGPUModelBuilder._return_model = _patched_return_model
    SingleGPUModelBuilder._pr23366_meta_patch_applied = True


def _to_probe_payload(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _to_probe_payload(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {key: _to_probe_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_probe_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_probe_payload(item) for item in value)
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, Path)):
        return {
            key: _to_probe_payload(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _save_probe(probe_dir: Path | None, relative_path: str, payload) -> None:
    if probe_dir is None:
        return
    output_path = probe_dir / f"{relative_path}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_to_probe_payload(payload), output_path)


def _selected_probe_block_indices(num_blocks: int) -> set[int]:
    if num_blocks <= 0:
        return set()
    last_idx = num_blocks - 1
    selected = {0, last_idx}
    for idx in (16, 24, 28, 32):
        if idx <= last_idx:
            selected.add(idx)
    return selected


def _probe_block_basename(phase: str, block_idx: int, num_blocks: int) -> str:
    last_idx = num_blocks - 1
    if block_idx == last_idx:
        return f"{phase}_transformer_block_last"
    return f"{phase}_transformer_block{block_idx}"


def _patch_stage2_block_probe_hooks() -> None:
    if getattr(BasicAVTransformerBlock, "_pr23366_block_probe_patch_applied", False):
        return

    original_forward = BasicAVTransformerBlock.forward
    original_apply_text_cross_attention = (
        BasicAVTransformerBlock._apply_text_cross_attention
    )

    def _wrapped_forward(self, video, audio, perturbations=None):
        state = _STAGE2_BLOCK_PROBE_STATE
        enabled = bool(state["enabled"])
        selected_blocks = state["selected_blocks"]
        if not enabled or self.idx not in selected_blocks:
            return original_forward(self, video, audio, perturbations)

        probe_dir = state["probe_dir"]
        phase = str(state["phase"])
        num_blocks = int(state["num_blocks"])
        block_basename = _probe_block_basename(phase, self.idx, num_blocks)
        if not bool(state.get("preprocessed_saved", False)):
            _save_probe(
                probe_dir,
                f"denoising/{phase}_transformer_preprocessed",
                {
                    "video": None
                    if video is None
                    else {
                        "x": video.x,
                        "context": video.context,
                        "context_mask": video.context_mask,
                        "timesteps": video.timesteps,
                        "embedded_timestep": video.embedded_timestep,
                        "prompt_timestep": video.prompt_timestep,
                    },
                    "audio": None
                    if audio is None
                    else {
                        "x": audio.x,
                        "context": audio.context,
                        "context_mask": audio.context_mask,
                        "timesteps": audio.timesteps,
                        "embedded_timestep": audio.embedded_timestep,
                        "prompt_timestep": audio.prompt_timestep,
                    },
                },
            )
            state["preprocessed_saved"] = True
        _save_probe(
            probe_dir,
            f"denoising/{block_basename}_input",
            {
                "video": None if video is None else video.x,
                "audio": None if audio is None else audio.x,
            },
        )
        outputs = original_forward(self, video, audio, perturbations)
        out_video, out_audio = outputs
        _save_probe(
            probe_dir,
            f"denoising/{block_basename}",
            {
                "video": None if out_video is None else out_video.x,
                "audio": None if out_audio is None else out_audio.x,
            },
        )
        return outputs

    def _wrapped_apply_text_cross_attention(
        self,
        x,
        context,
        attn,
        scale_shift_table,
        prompt_scale_shift_table,
        timestep,
        prompt_timestep,
        context_mask,
        cross_attention_adaln=False,
    ):
        out = original_apply_text_cross_attention(
            self,
            x,
            context,
            attn,
            scale_shift_table,
            prompt_scale_shift_table,
            timestep,
            prompt_timestep,
            context_mask,
            cross_attention_adaln=cross_attention_adaln,
        )

        state = _STAGE2_BLOCK_PROBE_STATE
        enabled = bool(state["enabled"])
        selected_blocks = state["selected_blocks"]
        if not enabled or self.idx not in selected_blocks:
            return out

        probe_dir = state["probe_dir"]
        phase = str(state["phase"])
        num_blocks = int(state["num_blocks"])
        block_basename = _probe_block_basename(phase, self.idx, num_blocks)
        prompt_call_counts = state["prompt_call_counts"]
        prompt_inputs = state["prompt_inputs"]
        prompt_deltas = state["prompt_deltas"]

        call_idx = int(prompt_call_counts.get(self.idx, 0))
        modality = "video" if call_idx % 2 == 0 else "audio"
        prompt_call_counts[self.idx] = call_idx + 1

        per_block_inputs = prompt_inputs.setdefault(self.idx, {})
        per_block_inputs[f"{modality}_attn_input"] = x
        per_block_inputs[f"{modality}_context"] = context
        per_block_inputs[f"{modality}_context_mask"] = context_mask
        _save_probe(
            probe_dir,
            f"denoising/{block_basename}_prompt_cross_inputs",
            per_block_inputs,
        )

        per_block_deltas = prompt_deltas.setdefault(self.idx, {})
        per_block_deltas[f"{modality}_delta"] = out
        _save_probe(
            probe_dir,
            f"denoising/{block_basename}_prompt_cross_delta",
            per_block_deltas,
        )
        return out

    BasicAVTransformerBlock.forward = _wrapped_forward
    BasicAVTransformerBlock._apply_text_cross_attention = (
        _wrapped_apply_text_cross_attention
    )
    BasicAVTransformerBlock._pr23366_block_probe_patch_applied = True


@contextmanager
def _active_stage2_block_probe(probe_dir: Path | None, transformer) -> None:
    if probe_dir is None:
        yield
        return

    velocity_model = getattr(transformer, "velocity_model", None)
    transformer_blocks = getattr(velocity_model, "transformer_blocks", None)
    if transformer_blocks is None:
        yield
        return

    num_blocks = len(transformer_blocks)
    _STAGE2_BLOCK_PROBE_STATE.update(
        {
            "enabled": True,
            "probe_dir": probe_dir,
            "phase": "stage2",
            "num_blocks": num_blocks,
            "selected_blocks": _selected_probe_block_indices(num_blocks),
            "prompt_call_counts": {},
            "prompt_inputs": {},
            "prompt_deltas": {},
            "preprocessed_saved": False,
        }
    )
    try:
        yield
    finally:
        _STAGE2_BLOCK_PROBE_STATE.update(
            {
                "enabled": False,
                "probe_dir": None,
                "num_blocks": 0,
                "selected_blocks": set(),
                "prompt_call_counts": {},
                "prompt_inputs": {},
                "prompt_deltas": {},
                "preprocessed_saved": False,
            }
        )


def _patch_stage2_probe_loop(probe_dir: Path | None):
    if probe_dir is None:
        return None

    _patch_stage2_block_probe_hooks()
    original_loop = ltx2_hq_module.res2s_audio_video_denoising_loop

    def _wrapped_loop(*, sigmas, video_state, audio_state, stepper, transformer, denoiser):
        is_stage2 = int(sigmas.numel()) <= 4
        if not is_stage2:
            video_state, audio_state = original_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                transformer=transformer,
                denoiser=denoiser,
            )
            _save_probe(
                probe_dir,
                "stage1/output",
                {
                    "video_state": video_state,
                    "audio_state": audio_state,
                },
            )
            return video_state, audio_state

        _save_probe(
            probe_dir,
            "stage2/noised_input",
            {
                "video_state": video_state,
                "audio_state": audio_state,
            },
        )
        did_probe_step0 = False

        def _wrapped_denoiser(*denoiser_args, **denoiser_kwargs):
            nonlocal did_probe_step0
            step_index = denoiser_kwargs.get("step_index")
            if step_index is None and len(denoiser_args) >= 5:
                step_index = denoiser_args[4]
            transformer_inner = denoiser_kwargs.get("transformer")
            if transformer_inner is None and denoiser_args:
                transformer_inner = denoiser_args[0]

            if step_index == 0 and not did_probe_step0:
                did_probe_step0 = True
                with _active_stage2_block_probe(probe_dir, transformer_inner):
                    return denoiser(*denoiser_args, **denoiser_kwargs)
            return denoiser(*denoiser_args, **denoiser_kwargs)

        video_state, audio_state = original_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=transformer,
            denoiser=_wrapped_denoiser,
        )
        _save_probe(
            probe_dir,
            "stage2/output",
            {
                "video_state": video_state,
                "audio_state": audio_state,
            },
        )
        return video_state, audio_state

    ltx2_hq_module.res2s_audio_video_denoising_loop = _wrapped_loop
    return original_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pinned official LTX-2.3 HQ two-stage reproduction for the SpongeBob "
            "storyboard case used in SGLang PR #23366."
        )
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--distilled-lora-path", required=True)
    parser.add_argument("--spatial-upsampler-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help=(
            "Pinned seed for stable reference generation. "
            "PR #23366 did not spell out a seed in the sample command, but this "
            "script defaults to 10 so the official reference is reproducible."
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Override the official default negative prompt if needed.",
    )
    parser.add_argument("--probe-dir")
    return parser.parse_args()


def _clone_video_chunks(chunks):
    for chunk in chunks:
        yield chunk.clone()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    _patch_official_builder_for_gemma_buffers()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe_dir).expanduser().resolve() if args.probe_dir else None
    if probe_dir is not None:
        probe_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TI2VidTwoStagesHQPipeline(
        checkpoint_path=str(Path(args.checkpoint_path).expanduser().resolve()),
        distilled_lora=[
            LoraPathStrengthAndSDOps(
                str(Path(args.distilled_lora_path).expanduser().resolve()),
                1.0,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ],
        distilled_lora_strength_stage_1=DISTILLED_LORA_STAGE_1,
        distilled_lora_strength_stage_2=DISTILLED_LORA_STAGE_2,
        spatial_upsampler_path=str(
            Path(args.spatial_upsampler_path).expanduser().resolve()
        ),
        gemma_root=str(Path(args.gemma_root).expanduser().resolve()),
        loras=(),
    )

    tiling_config = TilingConfig.default()
    original_stage2_loop = _patch_stage2_probe_loop(probe_dir)
    try:
        video, audio = pipeline(
            prompt=PR23366_SPONGEBOB_PROMPT,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            frame_rate=FRAME_RATE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            video_guider_params=VIDEO_GUIDER_PARAMS,
            audio_guider_params=AUDIO_GUIDER_PARAMS,
            images=[],
            tiling_config=tiling_config,
        )
    finally:
        if original_stage2_loop is not None:
            ltx2_hq_module.res2s_audio_video_denoising_loop = original_stage2_loop

    encode_video(
        video=_clone_video_chunks(video),
        fps=FRAME_RATE,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=get_video_chunks_number(NUM_FRAMES, tiling_config),
    )
    print(output_path)


if __name__ == "__main__":
    main()
