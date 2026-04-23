from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

try:
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.model.transformer.attention import PytorchAttention
    from ltx_core.text_encoders.gemma.embeddings_processor import (
        convert_to_additive_mask,
    )
    from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        EmbeddingsProcessorConfigurator,
    )
    from ltx_core.text_encoders.gemma.feature_extractor import (
        norm_and_concat_per_token_rms,
    )
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from transformers import Gemma3ForConditionalGeneration
except ImportError as exc:  # pragma: no cover - manual helper
    raise SystemExit(
        "This script must run against the official LTX-2 repo. "
        "Set PYTHONPATH to include packages/ltx-core/src."
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the official LTX-2.3 text path against a native SGLang "
            "probe dump. Covers Gemma hidden states, packed prompt embeds, "
            "feature extractor outputs, and connector outputs."
        )
    )
    parser.add_argument("--native-dir", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--official-attention", default="default", choices=("default", "pytorch"))
    parser.add_argument("--save-dir")
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_probe(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _tensor_psnr(reference: torch.Tensor, other: torch.Tensor) -> float:
    if reference.shape != other.shape:
        raise ValueError(f"Shape mismatch: {tuple(reference.shape)} vs {tuple(other.shape)}")
    ref = reference.detach().float().cpu()
    cur = other.detach().float().cpu()
    mse = torch.mean((ref - cur) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = max(ref.abs().max().item(), cur.abs().max().item(), 1e-12)
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def _shape(value: torch.Tensor | None) -> str:
    return "None" if value is None else str(tuple(value.shape))


def _report_tensor(name: str, native_value: torch.Tensor, official_value: torch.Tensor) -> None:
    psnr = _tensor_psnr(native_value, official_value)
    print(f"{name}: shape={tuple(native_value.shape)} psnr={psnr:.4f} dB")


def _resolve_gemma_roots(gemma_root: Path) -> tuple[Path, Path]:
    text_encoder_root = gemma_root / "text_encoder"
    tokenizer_root = gemma_root / "tokenizer"
    if text_encoder_root.is_dir() and tokenizer_root.is_dir():
        return text_encoder_root, tokenizer_root
    model_candidates = sorted(gemma_root.rglob("model*.safetensors"))
    tokenizer_candidates = sorted(gemma_root.rglob("tokenizer.model"))
    if not model_candidates or not tokenizer_candidates:
        raise FileNotFoundError(f"Could not resolve Gemma roots under {gemma_root}")
    return model_candidates[0].parent, tokenizer_candidates[0].parent


def _build_official_text_encoder(
    gemma_root: Path, device: torch.device, dtype: torch.dtype
) -> GemmaTextEncoder:
    text_encoder_root, tokenizer_root = _resolve_gemma_roots(gemma_root)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(text_encoder_root),
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    tokenizer = LTXVGemmaTokenizer(str(tokenizer_root), 1024)
    return GemmaTextEncoder(model=model, tokenizer=tokenizer, processor=None, dtype=dtype)


def _remap_componentized_connector_key_for_official(key: str) -> str:
    if key.startswith("video_aggregate_embed."):
        return "feature_extractor.video_aggregate_embed." + key[len("video_aggregate_embed.") :]
    if key.startswith("audio_aggregate_embed."):
        return "feature_extractor.audio_aggregate_embed." + key[len("audio_aggregate_embed.") :]
    if key.startswith("text_proj_in."):
        return "feature_extractor.aggregate_embed." + key[len("text_proj_in.") :]
    if key.startswith("video_connector."):
        suffix = key[len("video_connector.") :]
        suffix = suffix.replace("transformer_blocks", "transformer_1d_blocks")
        suffix = suffix.replace(".attn1.norm_q.", ".attn1.q_norm.")
        suffix = suffix.replace(".attn1.norm_k.", ".attn1.k_norm.")
        return f"video_connector.{suffix}"
    if key.startswith("audio_connector."):
        suffix = key[len("audio_connector.") :]
        suffix = suffix.replace("transformer_blocks", "transformer_1d_blocks")
        suffix = suffix.replace(".attn1.norm_q.", ".attn1.q_norm.")
        suffix = suffix.replace(".attn1.norm_k.", ".attn1.k_norm.")
        return f"audio_connector.{suffix}"
    return key


def _build_embeddings_processor(
    checkpoint_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    official_attention: str,
):
    transformer_config = json.loads(
        (checkpoint_root / "transformer" / "config.json").read_text()
    )
    connector_model_path = checkpoint_root / "connectors" / "model.safetensors"
    state_dict = SafetensorsModelStateDictLoader().load(
        [str(connector_model_path)],
        sd_ops=None,
        device=torch.device("cpu"),
    ).sd
    state_dict = {
        _remap_componentized_connector_key_for_official(key): value
        for key, value in state_dict.items()
    }
    model = EmbeddingsProcessorConfigurator.from_config({"transformer": transformer_config})
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            "official embeddings_processor load mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    if official_attention == "pytorch":
        for module in model.modules():
            if hasattr(module, "attention_function"):
                module.attention_function = PytorchAttention()
    return model.to(device=device, dtype=dtype).eval()


def _encode_single(
    text_encoder: GemmaTextEncoder,
    embeddings_processor,
    text: str,
) -> dict[str, torch.Tensor]:
    hidden_states, attention_mask = text_encoder.encode(text)
    stacked_hidden_states = torch.stack(list(hidden_states), dim=0)
    packed_hidden_states = norm_and_concat_per_token_rms(
        torch.stack(list(hidden_states), dim=-1),
        attention_mask,
    )
    video_hidden_states, audio_hidden_states = embeddings_processor.feature_extractor(
        hidden_states,
        attention_mask,
        padding_side="left",
    )
    additive_attention_mask = convert_to_additive_mask(
        attention_mask, video_hidden_states.dtype
    )
    video_prompt_embeds, audio_prompt_embeds, connector_mask = (
        embeddings_processor.create_embeddings(
            video_features=video_hidden_states,
            audio_features=audio_hidden_states,
            additive_attention_mask=additive_attention_mask,
        )
    )
    return {
        "input_ids": torch.tensor(
            [[t[0] for t in text_encoder.tokenizer.tokenize_with_weights(text)["gemma"]]],
            device=attention_mask.device,
        ),
        "attention_mask": attention_mask,
        "stacked_hidden_states": stacked_hidden_states,
        "packed_hidden_states": packed_hidden_states,
        "video_hidden_states": video_hidden_states,
        "audio_hidden_states": audio_hidden_states,
        "video_prompt_embeds": video_prompt_embeds,
        "audio_prompt_embeds": audio_prompt_embeds,
        "connector_mask": connector_mask,
    }


def _save_outputs(save_dir: Path, name: str, payload: dict[str, torch.Tensor]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in payload.items()}, save_dir / f"{name}.pt")


def main() -> None:
    args = _parse_args()
    native_dir = Path(args.native_dir).expanduser().resolve()
    gemma_root = Path(args.gemma_root).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    save_dir = None if args.save_dir is None else Path(args.save_dir).expanduser().resolve()
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)

    native_positive = _load_probe(native_dir / "text_encoding" / "positive" / "encoder_0.pt")
    native_negative = _load_probe(native_dir / "text_encoding" / "negative" / "encoder_0.pt")
    native_feature = _load_probe(native_dir / "text_connector" / "feature_extractor_output_split.pt")
    native_connector = _load_probe(native_dir / "text_connector" / "output_split.pt")

    prompt = native_positive["processed_text"][0]
    negative_prompt = native_negative["processed_text"][0]

    print(f"prompt chars={len(prompt)}")
    print(f"negative prompt chars={len(negative_prompt)}")

    with torch.inference_mode():
        text_encoder = _build_official_text_encoder(gemma_root, device, dtype)
        embeddings_processor = _build_embeddings_processor(
            checkpoint_root,
            device,
            dtype,
            args.official_attention,
        )

        official_positive = _encode_single(text_encoder, embeddings_processor, prompt)
        official_negative = _encode_single(text_encoder, embeddings_processor, negative_prompt)

    def _count_diffs(a: torch.Tensor, b: torch.Tensor) -> int:
        if a.shape != b.shape:
            return -1
        return int((a.cpu() != b.cpu()).sum().item())

    print(
        "positive input_ids diff_count="
        f"{_count_diffs(native_positive['input_ids'], official_positive['input_ids'])}"
    )
    print(
        "negative input_ids diff_count="
        f"{_count_diffs(native_negative['input_ids'], official_negative['input_ids'])}"
    )
    print(
        "positive attention_mask diff_count="
        f"{_count_diffs(native_positive['attention_mask'], official_positive['attention_mask'])}"
    )
    print(
        "negative attention_mask diff_count="
        f"{_count_diffs(native_negative['attention_mask'], official_negative['attention_mask'])}"
    )

    _report_tensor(
        "positive stacked_hidden_states",
        native_positive["stacked_hidden_states"],
        official_positive["stacked_hidden_states"],
    )
    _report_tensor(
        "negative stacked_hidden_states",
        native_negative["stacked_hidden_states"],
        official_negative["stacked_hidden_states"],
    )
    _report_tensor(
        "positive packed_hidden_states",
        native_positive["prompt_embeds"],
        official_positive["packed_hidden_states"],
    )
    _report_tensor(
        "negative packed_hidden_states",
        native_negative["prompt_embeds"],
        official_negative["packed_hidden_states"],
    )
    _report_tensor(
        "positive feature video",
        native_feature["positive_video_hidden_states"],
        official_positive["video_hidden_states"],
    )
    _report_tensor(
        "negative feature video",
        native_feature["negative_video_hidden_states"],
        official_negative["video_hidden_states"],
    )
    _report_tensor(
        "positive feature audio",
        native_feature["positive_audio_hidden_states"],
        official_positive["audio_hidden_states"],
    )
    _report_tensor(
        "negative feature audio",
        native_feature["negative_audio_hidden_states"],
        official_negative["audio_hidden_states"],
    )
    _report_tensor(
        "positive connector video",
        native_connector["positive_prompt_embeds"],
        official_positive["video_prompt_embeds"],
    )
    _report_tensor(
        "negative connector video",
        native_connector["negative_prompt_embeds"],
        official_negative["video_prompt_embeds"],
    )
    _report_tensor(
        "positive connector audio",
        native_connector["positive_audio_prompt_embeds"],
        official_positive["audio_prompt_embeds"],
    )
    _report_tensor(
        "negative connector audio",
        native_connector["negative_audio_prompt_embeds"],
        official_negative["audio_prompt_embeds"],
    )
    _report_tensor(
        "positive connector mask",
        native_connector["positive_attention_mask"].float(),
        official_positive["connector_mask"].float(),
    )
    _report_tensor(
        "negative connector mask",
        native_connector["negative_attention_mask"].float(),
        official_negative["connector_mask"].float(),
    )

    if save_dir is not None:
        _save_outputs(save_dir, "official_positive", official_positive)
        _save_outputs(save_dir, "official_negative", official_negative)
        print(f"saved official outputs to {save_dir}")

    del embeddings_processor
    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
