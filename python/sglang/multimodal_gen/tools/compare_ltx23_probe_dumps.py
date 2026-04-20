from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _flatten_probe(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if torch.is_tensor(value):
        result[prefix or "."] = value
        return result
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_probe(item, child))
        return result
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            result.update(_flatten_probe(item, child))
        return result
    result[prefix or "."] = value
    return result


def _compare_tensor(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    lhs_f = lhs.detach().cpu().to(torch.float32)
    rhs_f = rhs.detach().cpu().to(torch.float32)
    diff = lhs_f - rhs_f
    rmse = float(torch.sqrt(torch.mean(diff.square())).item())
    mae = float(torch.mean(diff.abs()).item())
    max_abs = float(torch.max(diff.abs()).item())
    cosine = None
    if lhs_f.numel() == rhs_f.numel() and lhs_f.numel() > 0:
        cosine = float(
            torch.nn.functional.cosine_similarity(
                lhs_f.reshape(1, -1), rhs_f.reshape(1, -1), dim=1
            ).item()
        )
    return {
        "lhs_shape": list(lhs.shape),
        "rhs_shape": list(rhs.shape),
        "dtype_match": str(lhs.dtype) == str(rhs.dtype),
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "cosine": cosine,
    }


def _compare_objects(lhs: Any, rhs: Any) -> dict[str, Any]:
    lhs_flat = _flatten_probe(lhs)
    rhs_flat = _flatten_probe(rhs)
    shared_keys = sorted(set(lhs_flat) & set(rhs_flat))
    lhs_only = sorted(set(lhs_flat) - set(rhs_flat))
    rhs_only = sorted(set(rhs_flat) - set(lhs_flat))
    comparisons: dict[str, Any] = {}
    for key in shared_keys:
        lhs_value = lhs_flat[key]
        rhs_value = rhs_flat[key]
        if torch.is_tensor(lhs_value) and torch.is_tensor(rhs_value):
            comparisons[key] = _compare_tensor(lhs_value, rhs_value)
        else:
            comparisons[key] = {
                "lhs_value": lhs_value,
                "rhs_value": rhs_value,
                "match": lhs_value == rhs_value,
            }
    return {
        "shared_keys": len(shared_keys),
        "lhs_only_keys": lhs_only,
        "rhs_only_keys": rhs_only,
        "comparisons": comparisons,
    }


def _load_path(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _compare_files(lhs: Path, rhs: Path) -> dict[str, Any]:
    return _compare_objects(_load_path(lhs), _load_path(rhs))


def _compare_dirs(lhs: Path, rhs: Path) -> dict[str, Any]:
    lhs_files = {
        path.relative_to(lhs).as_posix(): path for path in sorted(lhs.rglob("*.pt"))
    }
    rhs_files = {
        path.relative_to(rhs).as_posix(): path for path in sorted(rhs.rglob("*.pt"))
    }
    shared = sorted(set(lhs_files) & set(rhs_files))
    result = {
        "shared_files": shared,
        "lhs_only_files": sorted(set(lhs_files) - set(rhs_files)),
        "rhs_only_files": sorted(set(rhs_files) - set(lhs_files)),
        "file_comparisons": {},
    }
    for rel in shared:
        result["file_comparisons"][rel] = _compare_files(lhs_files[rel], rhs_files[rel])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fresh LTX-2.3 probe dumps saved with torch.save."
    )
    parser.add_argument("--lhs", required=True)
    parser.add_argument("--rhs", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lhs = Path(args.lhs)
    rhs = Path(args.rhs)
    if lhs.is_dir() != rhs.is_dir():
        raise ValueError("Both --lhs and --rhs must be either files or directories.")
    result = _compare_dirs(lhs, rhs) if lhs.is_dir() else _compare_files(lhs, rhs)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
