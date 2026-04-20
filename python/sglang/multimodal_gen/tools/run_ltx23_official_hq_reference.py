"""Run the official LTX-2.3 HQ pipeline with a safe Gemma text-encoder loader.

This helper stays on the official `TI2VidTwoStagesHQPipeline` path while
working around the current upstream Gemma meta-tensor initialization bug seen
in `ltx_pipelines.ti2vid_two_stages_hq`.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


def _read_prompt(args: argparse.Namespace) -> str:
    if bool(args.prompt) == bool(args.prompt_file):
        raise ValueError("Exactly one of --prompt or --prompt-file must be provided.")
    if args.prompt is not None:
        return args.prompt.strip()
    return Path(args.prompt_file).read_text(encoding="utf-8").strip()


def _prepend_official_repo_to_syspath(official_repo_root: Path) -> None:
    for rel_path in (
        "packages/ltx-core/src",
        "packages/ltx-pipelines/src",
        "packages/ltx-trainer/src",
    ):
        sys.path.insert(0, str(official_repo_root / rel_path))


def _clone_video_chunks(video: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    for chunk in video:
        yield chunk.clone()


def _resolve_gemma_component_roots(gemma_root: Path) -> tuple[Path, Path, Path]:
    text_encoder_root = gemma_root / "text_encoder"
    tokenizer_root = gemma_root / "tokenizer"
    if text_encoder_root.is_dir() and tokenizer_root.is_dir():
        return text_encoder_root, tokenizer_root, tokenizer_root
    model_candidates = sorted(gemma_root.rglob("model*.safetensors"))
    tokenizer_candidates = sorted(gemma_root.rglob("tokenizer.model"))
    processor_candidates = sorted(gemma_root.rglob("preprocessor_config.json"))
    if not model_candidates or not tokenizer_candidates or not processor_candidates:
        raise FileNotFoundError(
            f"Could not resolve Gemma component roots under {gemma_root}"
        )
    return (
        model_candidates[0].parent,
        tokenizer_candidates[0].parent,
        processor_candidates[0].parent,
    )


def _to_probe_payload(value: Any) -> Any:
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


def _maybe_dump_probe(probe_dir: Path | None, relative_path: str, payload: Any) -> None:
    if probe_dir is None:
        return
    output_path = probe_dir / f"{relative_path}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_to_probe_payload(payload), output_path)


class _Stage1TransformerProbe:
    def __init__(self, transformer: Any, probe_dir: Path | None):
        self._transformer = transformer
        self._probe_dir = probe_dir
        self._did_dump = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transformer, name)

    def __call__(self, *args, **kwargs):
        video = kwargs.get("video")
        audio = kwargs.get("audio")
        perturbations = kwargs.get("perturbations")
        model_video, model_audio = self._transformer(*args, **kwargs)
        if not self._did_dump:
            _maybe_dump_probe(
                self._probe_dir,
                "denoising/stage1_transformer_step0",
                {
                    "video": video,
                    "audio": audio,
                    "perturbations": perturbations,
                    "model_video": model_video,
                    "model_audio": model_audio,
                },
            )
            self._did_dump = True
        return model_video, model_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official LTX-2.3 HQ two-stage pipeline while forcing the "
            "Gemma text encoder to load via transformers.from_pretrained."
        )
    )
    parser.add_argument("--official-repo-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--distilled-lora-path", required=True)
    parser.add_argument("--spatial-upsampler-path", required=True)
    parser.add_argument(
        "--gemma-root",
        required=True,
        help=(
            "Path to the official Gemma snapshot root that contains "
            "`text_encoder/` and `tokenizer/`."
        ),
    )
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--distilled-lora-strength-stage-1", type=float, default=0.25)
    parser.add_argument("--distilled-lora-strength-stage-2", type=float, default=0.5)
    parser.add_argument("--video-cfg-guidance-scale", type=float, default=3.0)
    parser.add_argument("--video-stg-guidance-scale", type=float, default=0.0)
    parser.add_argument("--video-rescale-scale", type=float, default=0.45)
    parser.add_argument("--a2v-guidance-scale", type=float, default=3.0)
    parser.add_argument("--video-skip-step", type=int, default=0)
    parser.add_argument("--audio-cfg-guidance-scale", type=float, default=7.0)
    parser.add_argument("--audio-stg-guidance-scale", type=float, default=0.0)
    parser.add_argument("--audio-rescale-scale", type=float, default=1.0)
    parser.add_argument("--v2a-guidance-scale", type=float, default=3.0)
    parser.add_argument("--audio-skip-step", type=int, default=0)
    parser.add_argument(
        "--streaming-prefetch-count",
        type=int,
        default=1,
        help=(
            "Official block-layer streaming prefetch count. "
            "Use 1 on H100-class GPUs to keep the HQ helper under memory limits."
        ),
    )
    parser.add_argument("--vae-spatial-tile-size", type=int, default=512)
    parser.add_argument("--vae-spatial-tile-overlap", type=int, default=64)
    parser.add_argument("--vae-temporal-tile-size", type=int, default=64)
    parser.add_argument("--vae-temporal-tile-overlap", type=int, default=24)
    parser.add_argument("--probe-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = _read_prompt(args)
    official_repo_root = Path(args.official_repo_root)
    gemma_root = Path(args.gemma_root)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe_dir) if args.probe_dir else None
    if probe_dir is not None:
        probe_dir.mkdir(parents=True, exist_ok=True)

    _prepend_official_repo_to_syspath(official_repo_root)

    from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor

    from ltx_core.components.diffusion_steps import Res2sDiffusionStep
    from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import (
        SpatialTilingConfig,
        TemporalTilingConfig,
        TilingConfig,
        get_video_chunks_number,
    )
    from ltx_core.text_encoders.gemma.embeddings_processor import (
        EmbeddingsProcessorOutput,
        convert_to_additive_mask,
    )
    from ltx_core.text_encoders.gemma.feature_extractor import (
        norm_and_concat_per_token_rms,
    )
    from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from ltx_core.types import VideoLatentShape, VideoPixelShape
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
    from ltx_pipelines.utils.constants import (
        DEFAULT_NEGATIVE_PROMPT,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )
    from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
    from ltx_pipelines.utils.helpers import cleanup_memory, combined_image_conditionings
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop
    from ltx_pipelines.utils.blocks import gpu_model
    from ltx_pipelines.utils.types import ModalitySpec

    negative_prompt = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT

    gemma_model_root, tokenizer_root, processor_root = _resolve_gemma_component_roots(
        gemma_root
    )

    pipeline = TI2VidTwoStagesHQPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=[
            LoraPathStrengthAndSDOps(
                path=args.distilled_lora_path,
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ],
        distilled_lora_strength_stage_1=args.distilled_lora_strength_stage_1,
        distilled_lora_strength_stage_2=args.distilled_lora_strength_stage_2,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=str(gemma_root),
        loras=(),
        torch_compile=False,
    )

    @contextmanager
    def _hf_text_encoder_ctx(_: int | None) -> Iterator[GemmaTextEncoder]:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            str(gemma_model_root),
            local_files_only=True,
            dtype=pipeline.prompt_encoder._dtype,
            low_cpu_mem_usage=True,
        ).to(pipeline.prompt_encoder._device)
        tokenizer = LTXVGemmaTokenizer(str(tokenizer_root), 1024)
        processor = Gemma3Processor(
            image_processor=AutoImageProcessor.from_pretrained(
                str(processor_root), local_files_only=True
            ),
            tokenizer=tokenizer.tokenizer,
        )
        text_encoder = GemmaTextEncoder(
            model=model.eval(),
            tokenizer=tokenizer,
            processor=processor,
            dtype=pipeline.prompt_encoder._dtype,
        ).eval()
        try:
            yield text_encoder
        finally:
            del text_encoder
            del processor
            del tokenizer
            del model
            cleanup_memory()

    pipeline.prompt_encoder._text_encoder_ctx = _hf_text_encoder_ctx

    tiling_config = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=args.vae_spatial_tile_size,
            tile_overlap_in_pixels=args.vae_spatial_tile_overlap,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=args.vae_temporal_tile_size,
            tile_overlap_in_frames=args.vae_temporal_tile_overlap,
        ),
    )
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.video_cfg_guidance_scale,
        stg_scale=args.video_stg_guidance_scale,
        rescale_scale=args.video_rescale_scale,
        modality_scale=args.a2v_guidance_scale,
        skip_step=args.video_skip_step,
        stg_blocks=[],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=args.audio_cfg_guidance_scale,
        stg_scale=args.audio_stg_guidance_scale,
        rescale_scale=args.audio_rescale_scale,
        modality_scale=args.v2a_guidance_scale,
        skip_step=args.audio_skip_step,
        stg_blocks=[],
    )

    with torch.no_grad():
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        images: list[object] = []

        with pipeline.prompt_encoder._text_encoder_ctx(None) as text_encoder:
            raw_outputs = [text_encoder.encode(prompt), text_encoder.encode(negative_prompt)]

        pos_hidden_states, pos_attention_mask = raw_outputs[0]
        neg_hidden_states, neg_attention_mask = raw_outputs[1]
        pos_stacked_hidden_states = torch.stack(list(pos_hidden_states), dim=0)
        neg_stacked_hidden_states = torch.stack(list(neg_hidden_states), dim=0)
        pos_preconnector = norm_and_concat_per_token_rms(
            pos_stacked_hidden_states.permute(1, 2, 3, 0), pos_attention_mask
        ).to(pos_stacked_hidden_states.dtype)
        neg_preconnector = norm_and_concat_per_token_rms(
            neg_stacked_hidden_states.permute(1, 2, 3, 0), neg_attention_mask
        ).to(neg_stacked_hidden_states.dtype)
        _maybe_dump_probe(
            probe_dir,
            "text_encoding/positive/encoder_0",
            {
                "attention_mask": pos_attention_mask,
                "stacked_hidden_states": pos_stacked_hidden_states,
                "prompt_embeds": pos_preconnector,
            },
        )
        _maybe_dump_probe(
            probe_dir,
            "text_encoding/negative/encoder_0",
            {
                "attention_mask": neg_attention_mask,
                "stacked_hidden_states": neg_stacked_hidden_states,
                "prompt_embeds": neg_preconnector,
            },
        )
        _maybe_dump_probe(
            probe_dir,
            "text_connector/input_cfg",
            {
                "positive_prompt_embeds": pos_preconnector,
                "negative_prompt_embeds": neg_preconnector,
                "positive_attention_mask": pos_attention_mask,
                "negative_attention_mask": neg_attention_mask,
            },
        )
        with gpu_model(
            pipeline.prompt_encoder._embeddings_processor_builder.build(
                device=pipeline.device, dtype=pipeline.prompt_encoder._dtype
            ).to(pipeline.device).eval()
        ) as embeddings_processor:
            pos_video_feats, pos_audio_feats = embeddings_processor.feature_extractor(
                pos_hidden_states, pos_attention_mask, "left"
            )
            neg_video_feats, neg_audio_feats = embeddings_processor.feature_extractor(
                neg_hidden_states, neg_attention_mask, "left"
            )
            pos_video_encoding, pos_audio_encoding, pos_binary_mask = (
                embeddings_processor.create_embeddings(
                    pos_video_feats,
                    pos_audio_feats,
                    convert_to_additive_mask(pos_attention_mask, pos_video_feats.dtype),
                )
            )
            neg_video_encoding, neg_audio_encoding, neg_binary_mask = (
                embeddings_processor.create_embeddings(
                    neg_video_feats,
                    neg_audio_feats,
                    convert_to_additive_mask(neg_attention_mask, neg_video_feats.dtype),
                )
            )

        ctx_p = EmbeddingsProcessorOutput(
            pos_video_encoding, pos_audio_encoding, pos_binary_mask
        )
        ctx_n = EmbeddingsProcessorOutput(
            neg_video_encoding, neg_audio_encoding, neg_binary_mask
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
        _maybe_dump_probe(
            probe_dir,
            "text_connector/output_split",
            {
                "positive_prompt_embeds": v_context_p,
                "positive_audio_prompt_embeds": a_context_p,
                "negative_prompt_embeds": v_context_n,
                "negative_audio_prompt_embeds": a_context_n,
                "positive_attention_mask": ctx_p.attention_mask,
                "negative_attention_mask": ctx_n.attention_mask,
            },
        )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=args.num_frames,
            width=args.width // 2,
            height=args.height // 2,
            fps=args.frame_rate,
        )
        stage_1_conditionings = pipeline.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=pipeline.device,
            )
        )
        empty_latent = torch.empty(
            VideoLatentShape.from_pixel_shape(stage_1_output_shape).to_torch_shape()
        )
        stepper = Res2sDiffusionStep()
        sigmas = (
            LTX2Scheduler()
            .execute(latent=empty_latent, steps=args.num_inference_steps)
            .to(dtype=torch.float32, device=pipeline.device)
        )
        _maybe_dump_probe(
            probe_dir,
            "stage1/setup",
            {
                "sigmas": sigmas,
                "video_guider_params": video_guider_params,
                "audio_guider_params": audio_guider_params,
                "stage_1_output_shape": {
                    "batch": stage_1_output_shape.batch,
                    "frames": stage_1_output_shape.frames,
                    "width": stage_1_output_shape.width,
                    "height": stage_1_output_shape.height,
                    "fps": stage_1_output_shape.fps,
                },
            },
        )

        def stage1_probe_loop(**kwargs):
            kwargs["transformer"] = _Stage1TransformerProbe(
                kwargs["transformer"], probe_dir
            )
            return res2s_audio_video_denoising_loop(**kwargs)

        video_state, audio_state = pipeline.stage_1(
            denoiser=GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider=MultiModalGuider(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                ),
            ),
            sigmas=sigmas,
            noiser=noiser,
            stepper=stepper,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=args.num_frames,
            fps=args.frame_rate,
            video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=a_context_p),
            loop=stage1_probe_loop,
            streaming_prefetch_count=args.streaming_prefetch_count,
            max_batch_size=1,
        )
        _maybe_dump_probe(
            probe_dir,
            "stage1/output",
            {
                "video_state": video_state,
                "audio_state": audio_state,
            },
        )

        upscaled_video_latent = pipeline.upsampler(video_state.latent[:1])
        distilled_sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES, device=pipeline.device
        )
        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=args.num_frames,
            width=args.width,
            height=args.height,
            fps=args.frame_rate,
        )
        stage_2_conditionings = pipeline.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=pipeline.device,
            )
        )

        video_state, audio_state = pipeline.stage_2(
            denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
            sigmas=distilled_sigmas,
            noiser=noiser,
            stepper=stepper,
            width=args.width,
            height=args.height,
            frames=args.num_frames,
            fps=args.frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
            loop=res2s_audio_video_denoising_loop,
            streaming_prefetch_count=args.streaming_prefetch_count,
        )

        # The decode path only needs the final latents plus the video/audio decoders.
        # Release the large stage / prompt components first to keep H100-class cards
        # from OOMing during tiled VAE decode.
        del ctx_p, ctx_n
        del v_context_p, a_context_p, v_context_n, a_context_n
        del pos_video_encoding, pos_audio_encoding, pos_binary_mask
        del neg_video_encoding, neg_audio_encoding, neg_binary_mask
        del pos_video_feats, pos_audio_feats, neg_video_feats, neg_audio_feats
        del pos_hidden_states, pos_attention_mask
        del neg_hidden_states, neg_attention_mask
        del pos_stacked_hidden_states, neg_stacked_hidden_states
        del pos_preconnector, neg_preconnector
        del stage_1_conditionings, stage_2_conditionings
        del empty_latent, sigmas, distilled_sigmas
        del upscaled_video_latent, noiser, stepper
        del pipeline.stage_1, pipeline.stage_2
        del pipeline.prompt_encoder, pipeline.image_conditioner, pipeline.upsampler
        cleanup_memory()

        video = pipeline.video_decoder(
            video_state.latent.clone(), tiling_config, generator
        )
        audio = pipeline.audio_decoder(audio_state.latent.clone())

    encode_video(
        video=_clone_video_chunks(video),
        fps=args.frame_rate,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=get_video_chunks_number(args.num_frames, tiling_config),
    )

    print(output_path)


if __name__ == "__main__":
    main()
