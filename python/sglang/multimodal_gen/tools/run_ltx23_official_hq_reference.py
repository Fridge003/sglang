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
from types import MethodType
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


def _apply_interleaved_rotary_emb(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return x * cos + x_rotated * sin


def _apply_split_rotary_emb(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos, sin = freqs
    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        b = x.shape[0]
        _, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(
            f"Expected x.shape[-1] to be even for split rotary, got {last}."
        )
    r = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, r)
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]
    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)
    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(b, t, -1)
    return out.to(dtype=x_dtype)


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


def _unwrap_tensor_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"Expected tensor-like projection output, got {type(value)!r}")


def _build_attention_qkv_probe_payload(
    attn_module: Any,
    x: torch.Tensor,
    *,
    context: torch.Tensor | None = None,
    pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    context_ = x if context is None else context
    raw_q = _unwrap_tensor_output(attn_module.to_q(x))
    raw_k = _unwrap_tensor_output(attn_module.to_k(context_))
    raw_v = _unwrap_tensor_output(attn_module.to_v(context_))

    q = raw_q
    k = raw_k
    q_norm = getattr(attn_module, "q_norm", None)
    k_norm = getattr(attn_module, "k_norm", None)
    if q_norm is not None:
        q = q_norm(q)
    if k_norm is not None:
        k = k_norm(k)

    payload: dict[str, torch.Tensor] = {
        "q_proj": raw_q,
        "k_proj": raw_k,
        "v_proj": raw_v,
        "q_post_qk_norm": q,
        "k_post_qk_norm": k,
    }

    if pe is not None:
        cos, sin = pe
        k_cos, k_sin = pe if k_pe is None else k_pe
        if cos.dim() == 3:
            q = _apply_interleaved_rotary_emb(q, (cos, sin))
            k = _apply_interleaved_rotary_emb(k, (k_cos, k_sin))
        else:
            q = _apply_split_rotary_emb(q, (cos, sin))
            k = _apply_split_rotary_emb(k, (k_cos, k_sin))
        payload["q_post_rope"] = q
        payload["k_post_rope"] = k

    return payload


def _selected_probe_block_indices(num_blocks: int) -> list[int]:
    if num_blocks <= 0:
        return []
    last_idx = num_blocks - 1
    selected = {0, last_idx}
    for idx in (16, 24, 28, 32):
        if last_idx >= idx:
            selected.add(idx)
    return sorted(idx for idx in selected if 0 <= idx < num_blocks)


def _block_probe_relative_path(block_idx: int, num_blocks: int) -> str:
    last_idx = num_blocks - 1
    if block_idx == last_idx:
        return "denoising/stage1_transformer_block_last"
    return f"denoising/stage1_transformer_block{block_idx}"


class _Stage1TransformerProbe:
    def __init__(self, transformer: Any, probe_dir: Path | None):
        self._transformer = transformer
        self._probe_dir = probe_dir
        self._did_dump = False
        self._probe_counts: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transformer, name)

    def _resolve_velocity_model(self) -> Any | None:
        model = self._transformer
        seen: set[int] = set()
        while model is not None and id(model) not in seen:
            seen.add(id(model))
            velocity_model = getattr(model, "velocity_model", None)
            if velocity_model is not None:
                return velocity_model
            model = getattr(model, "_model", None)
        return None

    def _dump_call_indexed_probe(self, relative_path: str, payload: Any) -> None:
        if self._probe_dir is None:
            return
        call_idx = self._probe_counts.get(relative_path, 0)
        self._probe_counts[relative_path] = call_idx + 1
        if call_idx == 0:
            _maybe_dump_probe(self._probe_dir, relative_path, payload)
        _maybe_dump_probe(
            self._probe_dir,
            f"{relative_path}_call{call_idx}",
            payload,
        )

    def _install_block_wrapper(self, block: Any, relative_path: str) -> Any | None:
        if block is None:
            return None

        had_instance_forward = "forward" in block.__dict__
        original_instance_forward = block.__dict__.get("forward")
        probe = self

        def _wrapped_forward(
            wrapped_block: Any,
            video: Any | None,
            audio: Any | None,
            perturbations: Any | None = None,
        ) -> tuple[Any | None, Any | None]:
            from ltx_core.guidance.perturbations import (
                BatchedPerturbationConfig,
                PerturbationType,
            )
            from ltx_core.utils import rms_norm

            if video is None and audio is None:
                raise ValueError("At least one of video or audio must be provided")

            batch_size = (video or audio).x.shape[0]
            if perturbations is None:
                perturbations = BatchedPerturbationConfig.empty(batch_size)

            vx = video.x if video is not None else None
            ax = audio.x if audio is not None else None

            run_vx = video is not None and video.enabled and vx.numel() > 0
            run_ax = audio is not None and audio.enabled and ax.numel() > 0
            run_a2v = run_vx and (audio is not None and ax.numel() > 0)
            run_v2a = run_ax and (video is not None and vx.numel() > 0)
            norm_vx = None
            norm_ax = None

            probe._dump_call_indexed_probe(
                f"{relative_path}_input",
                {"video": vx, "audio": ax},
            )

            if run_vx:
                vshift_msa, vscale_msa, vgate_msa = wrapped_block.get_ada_values(
                    wrapped_block.scale_shift_table,
                    vx.shape[0],
                    video.timesteps,
                    slice(0, 3),
                )
                norm_vx = (
                    rms_norm(vx, eps=wrapped_block.norm_eps) * (1 + vscale_msa)
                    + vshift_msa
                )
                probe._dump_call_indexed_probe(
                    f"{relative_path}_video_self_attn_qkv",
                    _build_attention_qkv_probe_payload(
                        wrapped_block.attn1,
                        norm_vx,
                        pe=video.positional_embeddings,
                    ),
                )
                del vshift_msa, vscale_msa
                all_perturbed = perturbations.all_in_batch(
                    PerturbationType.SKIP_VIDEO_SELF_ATTN, wrapped_block.idx
                )
                none_perturbed = not perturbations.any_in_batch(
                    PerturbationType.SKIP_VIDEO_SELF_ATTN, wrapped_block.idx
                )
                v_mask = (
                    perturbations.mask_like(
                        PerturbationType.SKIP_VIDEO_SELF_ATTN,
                        wrapped_block.idx,
                        vx,
                    )
                    if not all_perturbed and not none_perturbed
                    else None
                )
                vx = (
                    vx
                    + wrapped_block.attn1(
                        norm_vx,
                        pe=video.positional_embeddings,
                        mask=video.self_attention_mask,
                        perturbation_mask=v_mask,
                        all_perturbed=all_perturbed,
                    )
                    * vgate_msa
                )
                del vgate_msa, v_mask

            if run_ax:
                ashift_msa, ascale_msa, agate_msa = wrapped_block.get_ada_values(
                    wrapped_block.audio_scale_shift_table,
                    ax.shape[0],
                    audio.timesteps,
                    slice(0, 3),
                )
                norm_ax = (
                    rms_norm(ax, eps=wrapped_block.norm_eps) * (1 + ascale_msa)
                    + ashift_msa
                )
                probe._dump_call_indexed_probe(
                    f"{relative_path}_pre_self_attn_norm",
                    {"video": norm_vx if run_vx else None, "audio": norm_ax},
                )
                probe._dump_call_indexed_probe(
                    f"{relative_path}_audio_self_attn_qkv",
                    _build_attention_qkv_probe_payload(
                        wrapped_block.audio_attn1,
                        norm_ax,
                        pe=audio.positional_embeddings,
                    ),
                )
                del ashift_msa, ascale_msa
                all_perturbed = perturbations.all_in_batch(
                    PerturbationType.SKIP_AUDIO_SELF_ATTN, wrapped_block.idx
                )
                none_perturbed = not perturbations.any_in_batch(
                    PerturbationType.SKIP_AUDIO_SELF_ATTN, wrapped_block.idx
                )
                a_mask = (
                    perturbations.mask_like(
                        PerturbationType.SKIP_AUDIO_SELF_ATTN,
                        wrapped_block.idx,
                        ax,
                    )
                    if not all_perturbed and not none_perturbed
                    else None
                )
                ax = (
                    ax
                    + wrapped_block.audio_attn1(
                        norm_ax,
                        pe=audio.positional_embeddings,
                        mask=audio.self_attention_mask,
                        perturbation_mask=a_mask,
                        all_perturbed=all_perturbed,
                    )
                    * agate_msa
                )
                del agate_msa, norm_ax, a_mask
                if norm_vx is not None:
                    del norm_vx
            elif run_vx:
                probe._dump_call_indexed_probe(
                    f"{relative_path}_pre_self_attn_norm",
                    {"video": norm_vx, "audio": None},
                )
                del norm_vx

            probe._dump_call_indexed_probe(
                f"{relative_path}_post_self_attn",
                {"video": vx, "audio": ax},
            )

            if run_vx:
                vx = vx + wrapped_block._apply_text_cross_attention(
                    vx,
                    video.context,
                    wrapped_block.attn2,
                    wrapped_block.scale_shift_table,
                    getattr(wrapped_block, "prompt_scale_shift_table", None),
                    video.timesteps,
                    video.prompt_timestep,
                    video.context_mask,
                    cross_attention_adaln=wrapped_block.cross_attention_adaln,
                )

            if run_ax:
                ax = ax + wrapped_block._apply_text_cross_attention(
                    ax,
                    audio.context,
                    wrapped_block.audio_attn2,
                    wrapped_block.audio_scale_shift_table,
                    getattr(wrapped_block, "audio_prompt_scale_shift_table", None),
                    audio.timesteps,
                    audio.prompt_timestep,
                    audio.context_mask,
                    cross_attention_adaln=wrapped_block.cross_attention_adaln,
                )

            probe._dump_call_indexed_probe(
                f"{relative_path}_post_prompt_cross_attn",
                {"video": vx, "audio": ax},
            )

            if run_a2v or run_v2a:
                vx_norm3 = rms_norm(vx, eps=wrapped_block.norm_eps)
                ax_norm3 = rms_norm(ax, eps=wrapped_block.norm_eps)

                if run_a2v and not perturbations.all_in_batch(
                    PerturbationType.SKIP_A2V_CROSS_ATTN, wrapped_block.idx
                ):
                    scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = (
                        wrapped_block.get_av_ca_ada_values(
                            wrapped_block.scale_shift_table_a2v_ca_video,
                            vx.shape[0],
                            video.cross_scale_shift_timestep,
                            video.cross_gate_timestep,
                            slice(0, 2),
                        )
                    )
                    vx_scaled = (
                        vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                    )
                    del scale_ca_video_a2v, shift_ca_video_a2v
                    scale_ca_audio_a2v, shift_ca_audio_a2v, _ = (
                        wrapped_block.get_av_ca_ada_values(
                            wrapped_block.scale_shift_table_a2v_ca_audio,
                            ax.shape[0],
                            audio.cross_scale_shift_timestep,
                            audio.cross_gate_timestep,
                            slice(0, 2),
                        )
                    )
                    ax_scaled = (
                        ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                    )
                    del scale_ca_audio_a2v, shift_ca_audio_a2v
                    a2v_mask = perturbations.mask_like(
                        PerturbationType.SKIP_A2V_CROSS_ATTN,
                        wrapped_block.idx,
                        vx,
                    )
                    vx = vx + (
                        wrapped_block.audio_to_video_attn(
                            vx_scaled,
                            context=ax_scaled,
                            pe=video.cross_positional_embeddings,
                            k_pe=audio.cross_positional_embeddings,
                        )
                        * gate_out_a2v
                        * a2v_mask
                    )
                    del gate_out_a2v, a2v_mask, vx_scaled, ax_scaled

                probe._dump_call_indexed_probe(
                    f"{relative_path}_post_a2v_cross_attn",
                    {"video": vx, "audio": ax},
                )

                if run_v2a and not perturbations.all_in_batch(
                    PerturbationType.SKIP_V2A_CROSS_ATTN, wrapped_block.idx
                ):
                    scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = (
                        wrapped_block.get_av_ca_ada_values(
                            wrapped_block.scale_shift_table_a2v_ca_audio,
                            ax.shape[0],
                            audio.cross_scale_shift_timestep,
                            audio.cross_gate_timestep,
                            slice(2, 4),
                        )
                    )
                    ax_scaled = (
                        ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                    )
                    del scale_ca_audio_v2a, shift_ca_audio_v2a
                    scale_ca_video_v2a, shift_ca_video_v2a, _ = (
                        wrapped_block.get_av_ca_ada_values(
                            wrapped_block.scale_shift_table_a2v_ca_video,
                            vx.shape[0],
                            video.cross_scale_shift_timestep,
                            video.cross_gate_timestep,
                            slice(2, 4),
                        )
                    )
                    vx_scaled = (
                        vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                    )
                    del scale_ca_video_v2a, shift_ca_video_v2a
                    v2a_mask = perturbations.mask_like(
                        PerturbationType.SKIP_V2A_CROSS_ATTN,
                        wrapped_block.idx,
                        ax,
                    )
                    ax = ax + (
                        wrapped_block.video_to_audio_attn(
                            ax_scaled,
                            context=vx_scaled,
                            pe=audio.cross_positional_embeddings,
                            k_pe=video.cross_positional_embeddings,
                        )
                        * gate_out_v2a
                        * v2a_mask
                    )
                    del gate_out_v2a, v2a_mask, ax_scaled, vx_scaled

                del vx_norm3, ax_norm3
            else:
                probe._dump_call_indexed_probe(
                    f"{relative_path}_post_a2v_cross_attn",
                    {"video": vx, "audio": ax},
                )

            probe._dump_call_indexed_probe(
                f"{relative_path}_post_v2a_cross_attn",
                {"video": vx, "audio": ax},
            )

            if run_vx:
                vshift_mlp, vscale_mlp, vgate_mlp = wrapped_block.get_ada_values(
                    wrapped_block.scale_shift_table,
                    vx.shape[0],
                    video.timesteps,
                    slice(3, 6),
                )
                vx_scaled = (
                    rms_norm(vx, eps=wrapped_block.norm_eps) * (1 + vscale_mlp)
                    + vshift_mlp
                )
                vx = vx + wrapped_block.ff(vx_scaled) * vgate_mlp
                del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

            if run_ax:
                ashift_mlp, ascale_mlp, agate_mlp = wrapped_block.get_ada_values(
                    wrapped_block.audio_scale_shift_table,
                    ax.shape[0],
                    audio.timesteps,
                    slice(3, 6),
                )
                ax_scaled = (
                    rms_norm(ax, eps=wrapped_block.norm_eps) * (1 + ascale_mlp)
                    + ashift_mlp
                )
                ax = ax + wrapped_block.audio_ff(ax_scaled) * agate_mlp
                del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

            probe._dump_call_indexed_probe(
                f"{relative_path}_post_ff",
                {"video": vx, "audio": ax},
            )

            video_out = dataclasses.replace(video, x=vx) if video is not None else None
            audio_out = dataclasses.replace(audio, x=ax) if audio is not None else None
            probe._dump_call_indexed_probe(
                relative_path,
                {"video": video_out, "audio": audio_out},
            )
            return video_out, audio_out

        block.forward = MethodType(_wrapped_forward, block)

        def _restore() -> None:
            if had_instance_forward:
                block.forward = original_instance_forward
            else:
                delattr(block, "forward")

        return _restore

    def _install_block_wrappers(self, velocity_model: Any) -> list[Any]:
        blocks = getattr(velocity_model, "transformer_blocks", None)
        if not blocks:
            return []
        restores = []
        for block_idx in _selected_probe_block_indices(len(blocks)):
            restore = self._install_block_wrapper(
                blocks[block_idx],
                _block_probe_relative_path(block_idx, len(blocks)),
            )
            if restore is not None:
                restores.append(restore)
        return restores

    def __call__(self, *args, **kwargs):
        video = kwargs.get("video")
        audio = kwargs.get("audio")
        perturbations = kwargs.get("perturbations")
        block_restores: list[Any] = []
        if not self._did_dump:
            velocity_model = self._resolve_velocity_model()
            if velocity_model is not None:
                block_restores = self._install_block_wrappers(velocity_model)
                _maybe_dump_probe(
                    self._probe_dir,
                    "denoising/stage1_transformer_preprocessed",
                    {
                        "video": (
                            None
                            if video is None
                            else velocity_model.video_args_preprocessor.prepare(
                                video, audio
                            )
                        ),
                        "audio": (
                            None
                            if audio is None
                            else velocity_model.audio_args_preprocessor.prepare(
                                audio, video
                            )
                        ),
                        "perturbations": perturbations,
                    },
                )
        try:
            model_video, model_audio = self._transformer(*args, **kwargs)
        finally:
            for restore in reversed(block_restores):
                restore()
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

        _maybe_dump_probe(
            probe_dir,
            "text_connector/feature_extractor_output_split",
            {
                "positive_video_hidden_states": pos_video_feats,
                "negative_video_hidden_states": neg_video_feats,
                "positive_audio_hidden_states": pos_audio_feats,
                "negative_audio_hidden_states": neg_audio_feats,
                "positive_attention_mask": pos_attention_mask,
                "negative_attention_mask": neg_attention_mask,
            },
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
