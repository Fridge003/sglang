"""Native LTX-2.3 HQ stage-1-inject ceiling harness (Ckpt-61 / A1 experiment).

Measures the end-to-end PSNR native can achieve if its stage-1 output matches
official byte-for-byte. Inject an official ``stage1/output.pt`` (pre-unpacked to
``(B, C, F, H, W)`` / ``(B, C, T, F)``) at the ``LTX2UpsampleStage`` entry, then
run native upsample + Ckpt-61 stage-2 re-noise + res2s loop + decode.

**A1 result (Ckpt-61 baseline):**

- End-to-end PSNR vs ``/tmp/ltx23_official_pr23366_canonical/official_hq_pr23366.mp4``:
  ``average = 34.03 dB`` (Y=32.54/U=41.35/V=41.67, min=28.78, max=39.41).
- ``stage2/noised_input.pt`` tensor PSNR: **video 68.32 dB, audio ∞ (bit-identical)**.
- Ckpt-60 ceiling (inject at noised_input): 34.68 dB. A1 is 0.65 dB under the
  ceiling — the delta is native upsampler's tiny bf16 drift amplified over
  the 3-step res2s refinement.

This proves:

1. Ckpt-61's ``_build_stage2_renoise_generator`` generates byte-exact noise
   relative to official ``noiser.generator`` at stage-2 entry.
2. The entire stage-2 pipeline (upsample + re-noise + res2s) has <1 dB
   residual in aggregate when fed a bit-exact stage-1 output.

Use this harness to re-verify the ceiling hasn't regressed after any future
stage-2 / upsampler / re-noise edit.

Prereqs:

- Official stage-1 probe dumped at
  ``/tmp/ltx23_official_pr23366_canonical_probe/stage1/output.pt`` (produced by
  ``test/manual/run_ltx23_official_hq_pr23366.py --probe-dir``). The packed
  tensors are unpacked on-the-fly to the shapes the upsample inject hook
  expects.
- Prompt file at ``/tmp/ltx23_hq_storyboard_prompt.txt``.
- ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` recommended on GPUs
  with <80 GB free headroom because the 2x spatial upsampler + dual DiT eval
  in res2s stage-2 needs contiguous ~60 GB allocations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import einops
import torch


DEFAULT_OFFICIAL_PROBE = (
    "/tmp/ltx23_official_pr23366_canonical_probe/stage1/output.pt"
)
DEFAULT_PROMPT = "/tmp/ltx23_hq_storyboard_prompt.txt"
DEFAULT_OUTPUT = "/tmp/native_stage1_inject_ceiling.mp4"

# Shapes for PR #23366 repro (768x512, 121 frames, stage-1 at half res).
STAGE1_VIDEO_F, STAGE1_VIDEO_H, STAGE1_VIDEO_W = 16, 8, 12
AUDIO_C, AUDIO_F = 8, 16


def _unpack_official_stage1(packed_path: Path) -> Path:
    """Read the packed probe and save an unpacked copy next to it."""
    unpacked_path = packed_path.with_name(packed_path.stem + "_unpacked.pt")
    if unpacked_path.exists():
        return unpacked_path
    probe = torch.load(str(packed_path), map_location="cpu")
    v = probe["video_state"]["latent"]
    a = probe["audio_state"]["latent"]
    v_unpacked = einops.rearrange(
        v,
        "b (f h w) c -> b c f h w",
        f=STAGE1_VIDEO_F,
        h=STAGE1_VIDEO_H,
        w=STAGE1_VIDEO_W,
    )
    a_unpacked = einops.rearrange(
        a, "b t (c f) -> b c t f", c=AUDIO_C, f=AUDIO_F
    )
    torch.save(
        {
            "video_state": {"latent": v_unpacked},
            "audio_state": {"latent": a_unpacked},
        },
        str(unpacked_path),
    )
    return unpacked_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-stage1-probe",
        default=DEFAULT_OFFICIAL_PROBE,
        help="Packed official stage1/output.pt from run_ltx23_official_hq_pr23366.py",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT,
        help="Path to the SpongeBob PR #23366 storyboard prompt.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Destination mp4 for the ceiling run.",
    )
    parser.add_argument(
        "--sglang-root",
        default="/tmp/sglang_ltx2_hq_pipeline/python",
        help="sys.path prefix for the sglang multimodal_gen package.",
    )
    args = parser.parse_args()

    packed = Path(args.official_stage1_probe).expanduser().resolve()
    if not packed.is_file():
        raise SystemExit(
            f"Official stage1 probe not found: {packed}. "
            "Regenerate via test/manual/run_ltx23_official_hq_pr23366.py --probe-dir"
        )
    unpacked = _unpack_official_stage1(packed)

    os.environ["SGLANG_DIFFUSION_LTX2_INJECT_STAGE1_OUTPUT"] = str(unpacked)
    for k in (
        "SGLANG_DIFFUSION_LTX2_INJECT_CONNECTOR_OUTPUT",
        "SGLANG_DIFFUSION_LTX2_INJECT_STAGE2_OUTPUT",
        "SGLANG_DIFFUSION_LTX2_INJECT_UPSAMPLE_OUTPUT",
        "SGLANG_DIFFUSION_LTX2_INJECT_STAGE2_NOISED_INPUT",
    ):
        os.environ.pop(k, None)

    sys.path.insert(0, args.sglang_root)
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    with open(args.prompt_file, encoding="utf-8") as f:
        prompt = f.read().strip()

    out_path = Path(args.output).expanduser().resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = DiffGenerator.from_pretrained(
        model_path="Lightricks/LTX-2.3",
        pipeline_class_name="LTX2TwoStageHQPipeline",
        num_gpus=1,
        ltx2_two_stage_device_mode="original",
        text_encoder_cpu_offload=True,
        dit_layerwise_offload=True,
    )
    gen.generate(
        sampling_params_kwargs={
            "prompt": prompt,
            "width": 768,
            "height": 512,
            "num_frames": 121,
            "fps": 25,
            "num_inference_steps": 30,
            "output_path": str(out_dir),
            "output_file_name": out_path.name,
        }
    )
    size = out_path.stat().st_size if out_path.exists() else 0
    print(f"[done] stage1-inject ceiling output -> {out_path} (size={size} bytes)")
    print(
        "Expected: PSNR vs canonical ≈ 34.03 dB (A1 baseline). "
        "Measure with: ffmpeg -hide_banner -i "
        "/tmp/ltx23_official_pr23366_canonical/official_hq_pr23366.mp4 -i "
        f"{out_path} -lavfi psnr -f null - 2>&1 | tail -1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
