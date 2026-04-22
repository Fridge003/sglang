"""Compare native vs official block-0 sub-layer probes for LTX-2.3 HQ.

Intended for Codex to run on GPU after:
  1. Patched official clone at /Users/mick/repos/sglang/3rdparty/LTX-2 (probe hook added by
     Claude in ltx_core/model/transformer/transformer.py) is synced to the remote host and
     imported via PYTHONPATH.
  2. Official run with:
         SGLANG_LTX_SUBLAYER_PROBE_DIR=/tmp/ltx23_official_sublayer_probe \
         SGLANG_LTX_SUBLAYER_PROBE_BLOCKS=0 \
         python3 /tmp/run_ltx23_official_hq_pr23366_stage1probe.py
     -> produces block00_post_self_attn.pt, block00_post_prompt_cross_attn.pt,
        block00_post_a2v_cross_attn.pt, block00_post_v2a_cross_attn.pt,
        block00_post_ff.pt under the probe dir.
  3. Native run with:
         SGLANG_DIFFUSION_PROBE_DIR=/tmp/ckpt62_sublayer_probe \
         python3 /tmp/run_ckpt61_step1.py
     -> under {SGLANG_DIFFUSION_PROBE_DIR}/<req_id>/stage1_transformer_block0_{post_*}.pt
        each contains {"video": tensor, "audio": tensor}.

Usage:
  python3 compare_block0_sublayer_probes.py \
      --native-dir /tmp/ckpt62_sublayer_probe/<req_id> \
      --official-dir /tmp/ltx23_official_sublayer_probe \
      [--slice-native cond]  # if native probe was taken during batched/split passes

Output: per-sub-layer video+audio PSNR and mean_abs delta, plus a drop-detector that
flags the step where audio PSNR drops below 50 dB (the expected elbow).
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch


SUBLAYER_NAMES = [
    "post_self_attn",
    "post_prompt_cross_attn",
    "post_a2v_cross_attn",
    "post_v2a_cross_attn",
    "post_ff",
]


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    if a.shape != b.shape:
        return float("nan")
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0.0:
        return float("inf")
    max_abs = max(a.abs().max().item(), b.abs().max().item())
    if max_abs == 0.0:
        return float("inf")
    return 20.0 * math.log10(max_abs / math.sqrt(mse))


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def _pick_pass(x: torch.Tensor, which: str, num_passes: int = 3) -> torch.Tensor:
    """Slice a batched probe. Native may have B = num_passes * 1 stacked from pass_specs
    in order ['cond', 'neg', 'modality']."""
    idx = {"cond": 0, "neg": 1, "modality": 2}[which]
    if x.shape[0] != num_passes:
        return x
    return x[idx : idx + 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-dir", required=True, help="e.g. /tmp/ckpt62_sublayer_probe/<req_id>")
    ap.add_argument("--official-dir", required=True, help="e.g. /tmp/ltx23_official_sublayer_probe")
    ap.add_argument(
        "--slice-native",
        default=None,
        choices=[None, "cond", "neg", "modality"],
        help="If native probe contains stacked batched passes, which one to compare.",
    )
    args = ap.parse_args()

    native_dir = Path(args.native_dir)
    official_dir = Path(args.official_dir)

    print(f"{'sub-layer':<28} {'V PSNR':>10} {'A PSNR':>10} {'V |Δ|':>12} {'A |Δ|':>12}")
    print("-" * 78)

    audio_elbow_reported = False
    for suffix in SUBLAYER_NAMES:
        native_path = native_dir / f"stage1_transformer_block0_{suffix}.pt"
        official_path = official_dir / f"block00_{suffix}.pt"
        if not native_path.exists() or not official_path.exists():
            print(f"{suffix:<28} MISSING  native={native_path.exists()} official={official_path.exists()}")
            continue
        nat = torch.load(native_path, map_location="cpu")
        off = torch.load(official_path, map_location="cpu")
        nat_v, nat_a = nat["video"], nat["audio"]
        off_v, off_a = off["video"], off["audio"]
        if args.slice_native:
            nat_v = _pick_pass(nat_v, args.slice_native)
            nat_a = _pick_pass(nat_a, args.slice_native)

        v_psnr = _psnr(nat_v, off_v)
        a_psnr = _psnr(nat_a, off_a)
        v_ma = _mean_abs(nat_v, off_v)
        a_ma = _mean_abs(nat_a, off_a)
        print(
            f"{suffix:<28} {v_psnr:>10.4f} {a_psnr:>10.4f} {v_ma:>12.6f} {a_ma:>12.6f}"
        )
        if a_psnr < 50.0 and not audio_elbow_reported:
            print(f"  -> AUDIO PSNR ELBOW at {suffix}: falls below 50 dB here")
            audio_elbow_reported = True

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
