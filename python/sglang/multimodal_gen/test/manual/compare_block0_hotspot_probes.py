"""Compare native vs official block-0 hotspot probes for LTX-2.3 HQ.

Intended for the V2A/audio-FF drill-down after applying
`block0_hotspot_probe.patch` to the official LTX-2 clone.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


FILE_PAIRS = [
    ("stage1_transformer_block0_prompt_cross_audio_qkv.pt", "block00_prompt_cross_audio_qkv.pt"),
    ("stage1_transformer_block0_v2a_inputs.pt", "block00_v2a_inputs.pt"),
    ("stage1_transformer_block0_v2a_attn_qkv.pt", "block00_v2a_attn_qkv.pt"),
    ("stage1_transformer_block0_v2a_delta.pt", "block00_v2a_delta.pt"),
    ("stage1_transformer_block0_audio_ff_inputs.pt", "block00_audio_ff_inputs.pt"),
    ("stage1_transformer_block0_audio_ff_internal.pt", "block00_audio_ff_internal.pt"),
    ("stage1_transformer_block0_audio_ff_delta.pt", "block00_audio_ff_delta.pt"),
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
    idx = {"cond": 0, "neg": 1, "modality": 2}[which]
    if x.ndim == 0 or x.shape[0] != num_passes:
        return x
    return x[idx : idx + 1]


def _format_metric(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-dir", required=True)
    ap.add_argument("--official-dir", required=True)
    ap.add_argument(
        "--slice-native",
        default=None,
        choices=[None, "cond", "neg", "modality"],
        help="If native payload uses stacked batched passes, pick one pass before compare.",
    )
    args = ap.parse_args()

    native_dir = Path(args.native_dir)
    official_dir = Path(args.official_dir)

    print(f"{'file:key':<64} {'shape':<22} {'PSNR':>10} {'|Δ|':>12}")
    print("-" * 112)

    for native_name, official_name in FILE_PAIRS:
        native_path = native_dir / native_name
        official_path = official_dir / official_name
        if not native_path.exists() or not official_path.exists():
            print(
                f"{native_name:<64} missing native={native_path.exists()} official={official_path.exists()}"
            )
            continue

        nat = torch.load(native_path, map_location="cpu")
        off = torch.load(official_path, map_location="cpu")
        shared_keys = sorted(set(nat.keys()) & set(off.keys()))
        if not shared_keys:
            print(f"{native_name:<64} no shared keys")
            continue

        for key in shared_keys:
            nat_tensor = nat[key]
            off_tensor = off[key]
            if not isinstance(nat_tensor, torch.Tensor) or not isinstance(
                off_tensor, torch.Tensor
            ):
                continue
            if args.slice_native:
                nat_tensor = _pick_pass(nat_tensor, args.slice_native)
            psnr = _psnr(nat_tensor, off_tensor)
            mean_abs = _mean_abs(nat_tensor, off_tensor)
            shape = "x".join(str(dim) for dim in nat_tensor.shape)
            print(
                f"{native_name}:{key:<64}"[:64]
                + f" {shape:<22} {_format_metric(psnr):>10} {mean_abs:>12.6f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
