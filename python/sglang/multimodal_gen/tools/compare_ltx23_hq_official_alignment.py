"""Run the official LTX-2.3 HQ two-stage reference and compare it with SGLang HQ runs.

This tool is intended for the LTX-2.3 HQ alignment workflow:

1. Run the official HQ reference via ``python -m ltx_pipelines.ti2vid_two_stages_hq``.
2. Run SGLang ``LTX2TwoStageHQPipeline`` on 1 GPU.
3. Run SGLang ``LTX2TwoStageHQPipeline`` on TP.
4. Save the three output videos and summarize pairwise metrics.

Example:

    python -m sglang.multimodal_gen.tools.compare_ltx23_hq_official_alignment \
        --official-repo-root /tmp/LTX-2-official \
        --checkpoint-path /root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/<snap>/ltx-2.3-22b-dev.safetensors \
        --distilled-lora-path /root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/<snap>/ltx-2.3-22b-distilled-lora-384.safetensors \
        --spatial-upsampler-path /root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/<snap>/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
        --gemma-root /root/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/<snap> \
        --prompt-file /tmp/ltx23_prompt.txt \
        --workdir /tmp/ltx23_hq_alignment
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency on validation boxes
    structural_similarity = None


def _read_prompt(args: argparse.Namespace) -> str:
    if bool(args.prompt) == bool(args.prompt_file):
        raise ValueError("Exactly one of --prompt or --prompt-file must be provided.")
    if args.prompt is not None:
        return args.prompt.strip()
    return Path(args.prompt_file).read_text(encoding="utf-8").strip()


def _prefixed_pythonpath(entries: list[str], existing: str | None) -> str:
    if existing:
        return os.pathsep.join([*entries, existing])
    return os.pathsep.join(entries)


def _run_command(
    cmd: list[str],
    env: dict[str, str],
    *,
    cwd: str | None = None,
) -> float:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"RUN: {printable}", flush=True)
    start = time.time()
    subprocess.run(cmd, check=True, env=env, cwd=cwd)
    return time.time() - start


def _read_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    return np.stack(frames, axis=0)


def _psnr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    diff = lhs.astype(np.float32) - rhs.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _ssim(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    if structural_similarity is None:
        return None
    return float(
        structural_similarity(lhs, rhs, channel_axis=2, data_range=255.0)
    )


def _compare_videos(
    reference_path: Path,
    candidate_path: Path,
    *,
    psnr_threshold: float,
) -> dict[str, Any]:
    reference = _read_video_rgb(reference_path)
    candidate = _read_video_rgb(candidate_path)
    compared_frames = min(len(reference), len(candidate))
    psnrs: list[float] = []
    maes: list[float] = []
    ssims: list[float] = []
    for idx in range(compared_frames):
        ref_frame = reference[idx]
        cand_frame = candidate[idx]
        psnrs.append(_psnr(ref_frame, cand_frame))
        maes.append(
            float(
                np.abs(
                    ref_frame.astype(np.float32) - cand_frame.astype(np.float32)
                ).mean()
            )
        )
        ssim_value = _ssim(ref_frame, cand_frame)
        if ssim_value is not None:
            ssims.append(ssim_value)
    return {
        "reference_path": str(reference_path),
        "candidate_path": str(candidate_path),
        "reference_frames": int(len(reference)),
        "candidate_frames": int(len(candidate)),
        "compared_frames": int(compared_frames),
        "mean_psnr": float(np.mean(psnrs)),
        "min_psnr": float(np.min(psnrs)),
        "max_psnr": float(np.max(psnrs)),
        "frame0_psnr": float(psnrs[0]),
        "frame_last_psnr": float(psnrs[-1]),
        "mean_mae": float(np.mean(maes)),
        "max_mae": float(np.max(maes)),
        "mean_ssim": float(np.mean(ssims)) if ssims else None,
        "psnr_threshold": float(psnr_threshold),
        "psnr_above_threshold_frames": int(sum(psnr >= psnr_threshold for psnr in psnrs)),
        "psnr_above_threshold_ratio": float(
            sum(psnr >= psnr_threshold for psnr in psnrs) / compared_frames
        ),
    }


def _find_latest_mp4(output_dir: Path) -> Path:
    mp4s = sorted(output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not mp4s:
        raise RuntimeError(f"no mp4 generated under {output_dir}")
    return mp4s[-1]


def _prepare_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _validate_official_repo_layout(official_repo_root: Path) -> list[str]:
    pythonpath_entries = [
        str(official_repo_root / "packages" / "ltx-core" / "src"),
        str(official_repo_root / "packages" / "ltx-pipelines" / "src"),
        str(official_repo_root / "packages" / "ltx-trainer" / "src"),
    ]
    missing = [entry for entry in pythonpath_entries if not Path(entry).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            "Official repo layout is incomplete. Missing expected PYTHONPATH entries: "
            f"{missing_text}"
        )
    return pythonpath_entries


def _is_hq_official_module(module_name: str) -> bool:
    return module_name.endswith("_hq")


def _build_official_module_args(
    args: argparse.Namespace, prompt: str, output_path: Path
) -> list[str]:
    module_args = [
        "--checkpoint-path",
        args.checkpoint_path,
        "--gemma-root",
        args.gemma_root,
        "--distilled-lora",
        args.distilled_lora_path,
        "--spatial-upsampler-path",
        args.spatial_upsampler_path,
        "--prompt",
        prompt,
        "--output-path",
        str(output_path),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-frames",
        str(args.num_frames),
        "--frame-rate",
        str(args.frame_rate),
        "--num-inference-steps",
        str(args.num_inference_steps),
    ]
    if _is_hq_official_module(args.official_module):
        module_args += [
            "--distilled-lora-strength-stage-1",
            str(args.official_distilled_lora_strength_stage_1),
            "--distilled-lora-strength-stage-2",
            str(args.official_distilled_lora_strength_stage_2),
        ]
    else:
        module_args.append(str(args.official_distilled_lora_strength))
    if args.seed is not None:
        module_args += ["--seed", str(args.seed)]
    return module_args


def _build_official_cmd(args: argparse.Namespace, prompt: str, output_path: Path) -> list[str]:
    module_args = _build_official_module_args(args, prompt, output_path)
    if args.official_use_uv:
        return [
            "uv",
            "run",
            "--project",
            args.official_repo_root,
            "--python",
            args.official_python_version,
            "python",
            "-m",
            args.official_module,
            *module_args,
        ]
    return [
        args.official_python_executable,
        "-m",
        args.official_module,
        *module_args,
    ]


def _build_sglang_cmd(
    args: argparse.Namespace,
    prompt: str,
    output_dir: Path,
    *,
    num_gpus: int,
    tp_size: int,
) -> list[str]:
    cmd = [
        "sglang",
        "generate",
        "--model-path",
        args.model_path,
        "--pipeline-class-name",
        args.pipeline_class_name,
        "--num-gpus",
        str(num_gpus),
        "--ltx2-two-stage-device-mode",
        args.ltx2_two_stage_device_mode,
        "--prompt",
        prompt,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--num-frames",
        str(args.num_frames),
        "--output-path",
        str(output_dir),
        "--text-encoder-cpu-offload",
        args.text_encoder_cpu_offload,
    ]
    if tp_size > 1:
        cmd += ["--tp-size", str(tp_size)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.enable_torch_compile:
        cmd.append("--enable-torch-compile")
    if args.warmup:
        cmd.append("--warmup")
    if args.dit_layerwise_offload:
        cmd.append("--dit-layerwise-offload")
    return cmd


def _run_official(
    args: argparse.Namespace,
    prompt: str,
    output_path: Path,
) -> tuple[float, list[str]]:
    official_repo_root = Path(args.official_repo_root)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.official_cuda_visible_devices
    if not args.official_use_uv:
        pythonpath_entries = _validate_official_repo_layout(official_repo_root)
        env["PYTHONPATH"] = _prefixed_pythonpath(
            pythonpath_entries, env.get("PYTHONPATH")
        )
    cmd = _build_official_cmd(args, prompt, output_path)
    elapsed = _run_command(cmd, env, cwd=str(official_repo_root))
    return elapsed, cmd


def _run_sglang_variant(
    args: argparse.Namespace,
    prompt: str,
    raw_output_dir: Path,
    stable_output_path: Path,
    *,
    num_gpus: int,
    tp_size: int,
    cuda_visible_devices: str,
) -> tuple[float, list[str]]:
    _prepare_dir(raw_output_dir)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    cmd = _build_sglang_cmd(
        args,
        prompt,
        raw_output_dir,
        num_gpus=num_gpus,
        tp_size=tp_size,
    )
    elapsed = _run_command(cmd, env)
    latest_mp4 = _find_latest_mp4(raw_output_dir)
    shutil.copy2(latest_mp4, stable_output_path)
    return elapsed, cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official LTX-2.3 two-stage reference and compare it against "
            "SGLang HQ 1GPU / TP outputs."
        )
    )
    parser.add_argument("--prompt", help="Prompt text to use for all three runs.")
    parser.add_argument(
        "--prompt-file",
        help="UTF-8 text file containing the full prompt.",
    )
    parser.add_argument(
        "--workdir",
        default="/tmp/ltx23_hq_alignment",
        help="Directory used to store videos, raw outputs, and metrics.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional explicit result JSON path. Defaults to <workdir>/result.json.",
    )
    parser.add_argument(
        "--official-repo-root",
        required=True,
        help="Root of the official LTX repo checkout containing packages/ltx-core and packages/ltx-pipelines.",
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--distilled-lora-path", required=True)
    parser.add_argument("--spatial-upsampler-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument(
        "--official-python-executable",
        default="python3",
        help="Python executable used for the plain official reference path when --official-use-uv is disabled.",
    )
    parser.add_argument(
        "--official-module",
        default="ltx_pipelines.ti2vid_two_stages_hq",
        help="Official module entrypoint used for the reference run.",
    )
    parser.add_argument(
        "--official-use-uv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the official reference through `uv run --project <official-repo-root>` "
            "instead of plain python."
        ),
    )
    parser.add_argument(
        "--official-python-version",
        default="3.11",
        help="Python version passed to `uv run --python` for the official reference.",
    )
    parser.add_argument(
        "--official-distilled-lora-strength",
        type=float,
        default=1.0,
        help="LoRA strength passed to the official two-stage script.",
    )
    parser.add_argument(
        "--official-distilled-lora-strength-stage-1",
        type=float,
        default=0.25,
        help="Stage-1 distilled LoRA strength for the official HQ reference.",
    )
    parser.add_argument(
        "--official-distilled-lora-strength-stage-2",
        type=float,
        default=0.5,
        help="Stage-2 distilled LoRA strength for the official HQ reference.",
    )
    parser.add_argument("--model-path", default="Lightricks/LTX-2.3")
    parser.add_argument(
        "--pipeline-class-name",
        default="LTX2TwoStageHQPipeline",
        help="SGLang pipeline class used for the candidate runs.",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=25.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--official-cuda-visible-devices",
        default="0",
        help="CUDA_VISIBLE_DEVICES used for the official reference run.",
    )
    parser.add_argument(
        "--sglang-1gpu-cuda-visible-devices",
        default="0",
        help="CUDA_VISIBLE_DEVICES used for the SGLang 1GPU run.",
    )
    parser.add_argument(
        "--sglang-tp-cuda-visible-devices",
        default="0,1,2,3",
        help="CUDA_VISIBLE_DEVICES used for the SGLang TP run.",
    )
    parser.add_argument(
        "--sglang-tp-size",
        type=int,
        default=4,
        help="Tensor parallel size for the TP run.",
    )
    parser.add_argument(
        "--ltx2-two-stage-device-mode",
        default="original",
        help="Value passed to --ltx2-two-stage-device-mode.",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to pass --enable-torch-compile to SGLang.",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to pass --warmup to SGLang.",
    )
    parser.add_argument(
        "--dit-layerwise-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pass --dit-layerwise-offload to SGLang.",
    )
    parser.add_argument(
        "--text-encoder-cpu-offload",
        choices=("true", "false"),
        default="true",
        help="Value passed to --text-encoder-cpu-offload for SGLang runs.",
    )
    parser.add_argument(
        "--psnr-threshold",
        type=float,
        default=35.0,
        help="Per-frame PSNR threshold used for the alignment ratio.",
    )
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help="Skip the official reference run.",
    )
    parser.add_argument(
        "--skip-sglang-1gpu",
        action="store_true",
        help="Skip the SGLang 1GPU run.",
    )
    parser.add_argument(
        "--skip-sglang-tp",
        action="store_true",
        help="Skip the SGLang TP run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = _read_prompt(args)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    official_output_path = workdir / "official.mp4"
    sglang_1gpu_raw_dir = workdir / "sglang_1gpu_raw"
    sglang_tp_raw_dir = workdir / f"sglang_tp{args.sglang_tp_size}_raw"
    sglang_1gpu_output_path = workdir / "sglang_1gpu.mp4"
    sglang_tp_output_path = workdir / f"sglang_tp{args.sglang_tp_size}.mp4"
    result_json_path = (
        Path(args.output_json) if args.output_json else workdir / "result.json"
    )

    videos: dict[str, Any] = {}
    commands: dict[str, list[str]] = {}
    metrics: dict[str, Any] = {}

    if not args.skip_official:
        official_elapsed, official_cmd = _run_official(
            args,
            prompt,
            official_output_path,
        )
        videos["official"] = {
            "path": str(official_output_path),
            "elapsed_sec": official_elapsed,
        }
        commands["official"] = official_cmd

    if not args.skip_sglang_1gpu:
        sglang_1gpu_elapsed, sglang_1gpu_cmd = _run_sglang_variant(
            args,
            prompt,
            sglang_1gpu_raw_dir,
            sglang_1gpu_output_path,
            num_gpus=1,
            tp_size=1,
            cuda_visible_devices=args.sglang_1gpu_cuda_visible_devices,
        )
        videos["sglang_1gpu"] = {
            "path": str(sglang_1gpu_output_path),
            "elapsed_sec": sglang_1gpu_elapsed,
            "raw_output_dir": str(sglang_1gpu_raw_dir),
        }
        commands["sglang_1gpu"] = sglang_1gpu_cmd

    if not args.skip_sglang_tp:
        sglang_tp_elapsed, sglang_tp_cmd = _run_sglang_variant(
            args,
            prompt,
            sglang_tp_raw_dir,
            sglang_tp_output_path,
            num_gpus=args.sglang_tp_size,
            tp_size=args.sglang_tp_size,
            cuda_visible_devices=args.sglang_tp_cuda_visible_devices,
        )
        videos[f"sglang_tp{args.sglang_tp_size}"] = {
            "path": str(sglang_tp_output_path),
            "elapsed_sec": sglang_tp_elapsed,
            "raw_output_dir": str(sglang_tp_raw_dir),
        }
        commands[f"sglang_tp{args.sglang_tp_size}"] = sglang_tp_cmd

    if not args.skip_official and not args.skip_sglang_1gpu:
        metrics["official_vs_sglang_1gpu"] = _compare_videos(
            official_output_path,
            sglang_1gpu_output_path,
            psnr_threshold=args.psnr_threshold,
        )
    if not args.skip_official and not args.skip_sglang_tp:
        metrics[f"official_vs_sglang_tp{args.sglang_tp_size}"] = _compare_videos(
            official_output_path,
            sglang_tp_output_path,
            psnr_threshold=args.psnr_threshold,
        )
    if not args.skip_sglang_1gpu and not args.skip_sglang_tp:
        metrics[f"sglang_1gpu_vs_sglang_tp{args.sglang_tp_size}"] = _compare_videos(
            sglang_1gpu_output_path,
            sglang_tp_output_path,
            psnr_threshold=args.psnr_threshold,
        )

    result = {
        "prompt": prompt,
        "config": {
            "official_repo_root": args.official_repo_root,
            "checkpoint_path": args.checkpoint_path,
            "distilled_lora_path": args.distilled_lora_path,
            "spatial_upsampler_path": args.spatial_upsampler_path,
            "gemma_root": args.gemma_root,
            "model_path": args.model_path,
            "pipeline_class_name": args.pipeline_class_name,
            "width": args.width,
            "height": args.height,
            "num_frames": args.num_frames,
            "frame_rate": args.frame_rate,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "psnr_threshold": args.psnr_threshold,
            "skip_official": args.skip_official,
            "skip_sglang_1gpu": args.skip_sglang_1gpu,
            "skip_sglang_tp": args.skip_sglang_tp,
        },
        "videos": videos,
        "commands": commands,
        "metrics": metrics,
        "ssim_available": structural_similarity is not None,
    }

    result_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"RESULT_JSON={result_json_path}", flush=True)


if __name__ == "__main__":
    main()
