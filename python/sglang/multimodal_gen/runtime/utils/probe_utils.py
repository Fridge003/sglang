from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sglang.multimodal_gen import envs


def _sanitize_probe_component(value: str | None) -> str:
    if not value:
        return "unknown"
    cleaned = []
    for ch in value:
        cleaned.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    sanitized = "".join(cleaned).strip("._")
    return sanitized or "unknown"


def _to_cpu_probe_payload(payload: Any) -> Any:
    if torch.is_tensor(payload):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        return {key: _to_cpu_probe_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_to_cpu_probe_payload(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_to_cpu_probe_payload(value) for value in payload)
    return payload


def get_probe_request_dir(batch: Any) -> Path | None:
    root = envs.SGLANG_DIFFUSION_PROBE_DIR
    if not root:
        return None

    request_id = _sanitize_probe_component(getattr(batch, "request_id", None))
    label = None
    extra = getattr(batch, "extra", None)
    if isinstance(extra, dict):
        label = extra.get("alignment_probe_label")

    dirname = request_id
    if label:
        dirname = f"{_sanitize_probe_component(str(label))}__{request_id}"

    request_dir = Path(root).expanduser() / dirname
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def dump_probe_payload(batch: Any, relative_path: str, payload: Any) -> Path | None:
    request_dir = get_probe_request_dir(batch)
    if request_dir is None:
        return None

    output_path = request_dir / f"{relative_path}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_to_cpu_probe_payload(payload), output_path)
    return output_path
