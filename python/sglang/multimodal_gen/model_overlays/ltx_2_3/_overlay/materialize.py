import json
import os

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.utils.model_overlay import (
    _copytree_link_or_copy,
    _ensure_dir,
    _link_or_copy_file,
)

AUXILIARY_MODEL_ID = "Lightricks/LTX-2"
AUXILIARY_MODEL_REVISION = "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
CONFIG_DONOR_MODEL_ID = "FastVideo/LTX-2.3-Distilled-Diffusers"
CONFIG_DONOR_MODEL_REVISION = "22b09fb1860a944bf10fa21f033d957d9ab9ec20"

AUXILIARY_PATTERNS = [
    "audio_vae/**",
    "scheduler/**",
    "text_encoder/**",
    "tokenizer/**",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

CONFIG_DONOR_PATTERNS = [
    "transformer/config.json",
    "text_encoder/config.json",
    "vae/**",
    "vocoder/**",
]

MONOLITH_PREFIX = "model.diffusion_model."
VIDEO_CONNECTOR_PREFIX = f"{MONOLITH_PREFIX}video_embeddings_connector."
AUDIO_CONNECTOR_PREFIX = f"{MONOLITH_PREFIX}audio_embeddings_connector."
TEXT_PROJ_IN_PREFIX = f"{MONOLITH_PREFIX}text_proj_in."
VIDEO_AGGREGATE_PREFIX = "text_embedding_projection.video_aggregate_embed."
AUDIO_AGGREGATE_PREFIX = "text_embedding_projection.audio_aggregate_embed."


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_checkpoint_config(path: str) -> dict:
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
    if metadata is None or "config" not in metadata:
        raise ValueError(f"Missing config metadata in checkpoint: {path}")
    return json.loads(metadata["config"])


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _build_text_encoder_config() -> dict:
    return {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "boi_token_index": 255999,
        "eoi_token_index": 256000,
        "eos_token_id": [1, 106],
        "image_token_id": 262144,
        "image_token_index": 262144,
        "initializer_range": 0.02,
        "mm_tokens_per_image": 256,
        "model_type": "gemma3",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_logit_softcapping": None,
            "cache_implementation": "hybrid",
            "final_logit_softcapping": None,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 3840,
            "initializer_range": 0.02,
            "intermediate_size": 15360,
            "max_position_embeddings": 131072,
            "model_type": "gemma3_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "num_key_value_heads": 8,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-6,
            "rope_local_base_freq": 10000,
            "rope_scaling": {
                "factor": 8.0,
                "rope_type": "linear",
            },
            "rope_theta": 1000000,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "torch_dtype": "float32",
            "use_cache": True,
            "vocab_size": 262208,
        },
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "vision_config": {
            "attention_dropout": 0.0,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "image_size": 896,
            "intermediate_size": 4304,
            "layer_norm_eps": 1e-6,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 27,
            "patch_size": 14,
            "torch_dtype": "float32",
            "vision_use_head": False,
        },
    }


def _rename_connector_key(key: str) -> str | None:
    if key.startswith(VIDEO_CONNECTOR_PREFIX):
        suffix = key[len(VIDEO_CONNECTOR_PREFIX) :]
        suffix = suffix.replace("transformer_1d_blocks", "transformer_blocks")
        suffix = suffix.replace(".attn1.q_norm.", ".attn1.norm_q.")
        suffix = suffix.replace(".attn1.k_norm.", ".attn1.norm_k.")
        return f"video_connector.{suffix}"
    if key.startswith(AUDIO_CONNECTOR_PREFIX):
        suffix = key[len(AUDIO_CONNECTOR_PREFIX) :]
        suffix = suffix.replace("transformer_1d_blocks", "transformer_blocks")
        suffix = suffix.replace(".attn1.q_norm.", ".attn1.norm_q.")
        suffix = suffix.replace(".attn1.k_norm.", ".attn1.norm_k.")
        return f"audio_connector.{suffix}"
    if key.startswith(TEXT_PROJ_IN_PREFIX):
        return key[len(MONOLITH_PREFIX) :]
    if key.startswith(VIDEO_AGGREGATE_PREFIX):
        return f"video_aggregate_embed.{key[len(VIDEO_AGGREGATE_PREFIX):]}"
    if key.startswith(AUDIO_AGGREGATE_PREFIX):
        return f"audio_aggregate_embed.{key[len(AUDIO_AGGREGATE_PREFIX):]}"
    return None


def _repack_transformer_weights(source_path: str, output_path: str) -> None:
    tensors = {}
    with safe_open(source_path, framework="pt") as f:
        for key in f.keys():
            if not key.startswith(MONOLITH_PREFIX):
                continue
            if key.startswith(VIDEO_CONNECTOR_PREFIX):
                continue
            if key.startswith(AUDIO_CONNECTOR_PREFIX):
                continue
            if key.startswith(TEXT_PROJ_IN_PREFIX):
                continue
            tensors[key[len(MONOLITH_PREFIX) :]] = f.get_tensor(key)
    if not tensors:
        raise ValueError("No transformer tensors found in LTX-2.3 source checkpoint.")
    save_file(tensors, output_path)


def _repack_connectors_weights(source_path: str, output_path: str) -> None:
    tensors = {}
    with safe_open(source_path, framework="pt") as f:
        for key in f.keys():
            renamed = _rename_connector_key(key)
            if renamed is None:
                continue
            tensors[renamed] = f.get_tensor(key)
    if not tensors:
        raise ValueError("No connector tensors found in LTX-2.3 source checkpoint.")
    save_file(tensors, output_path)


def _build_transformer_config(
    config_donor_dir: str, checkpoint_transformer_config: dict
) -> dict:
    config = _load_json(os.path.join(config_donor_dir, "transformer", "config.json"))
    config.update(checkpoint_transformer_config)
    config["_class_name"] = "LTX2VideoTransformer3DModel"
    config["ltx_variant"] = "ltx_2_3"
    config["cross_attention_adaln"] = True
    config["force_sdpa_v2a_cross_attention"] = True
    config["quantize_video_rope_coords_to_hidden_dtype"] = True
    if "double_precision_rope" not in config:
        config["double_precision_rope"] = (
            config.get("frequencies_precision") == "float64"
        )
    return config


def _build_connectors_config(
    config_donor_dir: str, checkpoint_transformer_config: dict
) -> dict:
    transformer_config = _load_json(
        os.path.join(config_donor_dir, "transformer", "config.json")
    )
    transformer_config.update(checkpoint_transformer_config)
    text_encoder_config = _load_json(
        os.path.join(config_donor_dir, "text_encoder", "config.json")
    )
    connector_max_pos = transformer_config.get("connector_positional_embedding_max_pos")
    connector_rope_base_seq_len = (
        1 if connector_max_pos is None else connector_max_pos[0]
    )
    connector_rope_type = transformer_config.get("rope_type", "interleaved")
    rope_double_precision = bool(
        transformer_config.get(
            "double_precision_rope",
            transformer_config.get("frequencies_precision", False) == "float64",
        )
    )
    video_connector_num_attention_heads = transformer_config.get(
        "connector_num_attention_heads", 30
    )
    video_connector_attention_head_dim = transformer_config.get(
        "connector_attention_head_dim", 128
    )
    video_connector_num_layers = transformer_config.get("connector_num_layers", 2)
    audio_connector_num_attention_heads = transformer_config.get(
        "audio_connector_num_attention_heads",
        video_connector_num_attention_heads,
    )
    audio_connector_attention_head_dim = transformer_config.get(
        "audio_connector_attention_head_dim",
        video_connector_attention_head_dim,
    )
    audio_connector_num_layers = transformer_config.get(
        "audio_connector_num_layers",
        video_connector_num_layers,
    )
    connector_apply_gated_attention = text_encoder_config.get(
        "connector_apply_gated_attention",
        transformer_config.get("connector_apply_gated_attention", False),
    )
    return {
        "_class_name": "LTX2TextConnectors",
        "_diffusers_version": "0.37.0.dev0",
        "audio_connector_attention_head_dim": audio_connector_attention_head_dim,
        "audio_connector_num_attention_heads": audio_connector_num_attention_heads,
        "audio_connector_num_layers": audio_connector_num_layers,
        "audio_connector_num_learnable_registers": text_encoder_config[
            "connector_num_learnable_registers"
        ],
        "audio_feature_extractor_out_features": text_encoder_config[
            "audio_feature_extractor_out_features"
        ],
        "caption_channels": text_encoder_config["hidden_size"],
        "causal_temporal_positioning": False,
        "connector_apply_gated_attention": connector_apply_gated_attention,
        "feature_extractor_in_features": text_encoder_config[
            "feature_extractor_in_features"
        ],
        "connector_rope_base_seq_len": connector_rope_base_seq_len,
        "rope_double_precision": rope_double_precision,
        "rope_theta": text_encoder_config["connector_positional_embedding_theta"],
        "rope_type": connector_rope_type,
        "text_proj_in_factor": text_encoder_config["feature_extractor_in_features"]
        // text_encoder_config["hidden_size"],
        "video_feature_extractor_out_features": text_encoder_config[
            "video_feature_extractor_out_features"
        ],
        "video_connector_attention_head_dim": video_connector_attention_head_dim,
        "video_connector_num_attention_heads": video_connector_num_attention_heads,
        "video_connector_num_layers": video_connector_num_layers,
        "video_connector_num_learnable_registers": text_encoder_config[
            "connector_num_learnable_registers"
        ],
    }


def _build_vae_config(auxiliary_dir: str, config_donor_dir: str) -> dict:
    config = _load_json(os.path.join(auxiliary_dir, "vae", "config.json"))
    config["ltx_variant"] = "ltx_2_3"
    config["condition_encoder_subdir"] = "ltx23_image_encoder"
    config["video_decoder_variant"] = "ltx_2_3"
    config["video_decoder_config"] = _load_json(
        os.path.join(config_donor_dir, "vae", "config.json")
    )["vae"]
    return config


def _repack_ltx23_image_encoder_weights(source_path: str, output_path: str) -> None:
    tensors = {}
    with safe_open(source_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith("encoder."):
                tensors[key[len("encoder.") :]] = f.get_tensor(key)
                continue
            if key.startswith("per_channel_statistics."):
                tensors[key] = f.get_tensor(key)
    if not tensors:
        raise ValueError("No LTX-2.3 image-encoder tensors found in donor checkpoint.")
    save_file(tensors, output_path)


def _repack_ltx23_video_decoder_weights(
    auxiliary_encoder_path: str,
    donor_decoder_path: str,
    output_path: str,
) -> None:
    tensors = {}
    with safe_open(auxiliary_encoder_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith("encoder."):
                tensors[key] = f.get_tensor(key)
    with safe_open(donor_decoder_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith("decoder."):
                tensors[key] = f.get_tensor(key)
                continue
            if key == "per_channel_statistics.mean-of-means":
                tensor = f.get_tensor(key)
                tensors["decoder.per_channel_statistics.mean_of_means"] = tensor
                tensors["latents_mean"] = tensor.clone()
                continue
            if key == "per_channel_statistics.std-of-means":
                tensor = f.get_tensor(key)
                tensors["decoder.per_channel_statistics.std_of_means"] = tensor
                tensors["latents_std"] = tensor.clone()
                continue
    if not tensors:
        raise ValueError("No LTX-2.3 decoder tensors found in donor checkpoint.")
    save_file(tensors, output_path)


def materialize(
    *,
    overlay_dir: str,
    source_dir: str,
    output_dir: str,
    manifest: dict,
) -> None:
    _ = overlay_dir, manifest

    auxiliary_dir = snapshot_download(
        repo_id=AUXILIARY_MODEL_ID,
        revision=AUXILIARY_MODEL_REVISION,
        allow_patterns=AUXILIARY_PATTERNS,
        max_workers=8,
    )
    config_donor_dir = snapshot_download(
        repo_id=CONFIG_DONOR_MODEL_ID,
        revision=CONFIG_DONOR_MODEL_REVISION,
        allow_patterns=CONFIG_DONOR_PATTERNS,
        max_workers=8,
    )

    for component_name in ("audio_vae", "scheduler", "text_encoder", "tokenizer"):
        _copytree_link_or_copy(
            os.path.join(auxiliary_dir, component_name),
            os.path.join(output_dir, component_name),
        )
    _write_json(
        os.path.join(output_dir, "text_encoder", "config.json"),
        _build_text_encoder_config(),
    )
    _copytree_link_or_copy(
        os.path.join(config_donor_dir, "vocoder"),
        os.path.join(output_dir, "vocoder"),
    )

    source_checkpoint = os.path.join(source_dir, "ltx-2.3-22b-dev.safetensors")
    checkpoint_transformer_config = _load_checkpoint_config(source_checkpoint).get(
        "transformer", {}
    )

    transformer_dir = os.path.join(output_dir, "transformer")
    _ensure_dir(transformer_dir)
    _write_json(
        os.path.join(transformer_dir, "config.json"),
        _build_transformer_config(config_donor_dir, checkpoint_transformer_config),
    )
    _repack_transformer_weights(
        source_checkpoint, os.path.join(transformer_dir, "model.safetensors")
    )

    connectors_dir = os.path.join(output_dir, "connectors")
    _ensure_dir(connectors_dir)
    _write_json(
        os.path.join(connectors_dir, "config.json"),
        _build_connectors_config(config_donor_dir, checkpoint_transformer_config),
    )
    _repack_connectors_weights(
        source_checkpoint, os.path.join(connectors_dir, "model.safetensors")
    )

    vae_dir = os.path.join(output_dir, "vae")
    _ensure_dir(vae_dir)
    _write_json(
        os.path.join(vae_dir, "config.json"),
        _build_vae_config(auxiliary_dir, config_donor_dir),
    )
    _repack_ltx23_video_decoder_weights(
        os.path.join(auxiliary_dir, "vae", "diffusion_pytorch_model.safetensors"),
        os.path.join(config_donor_dir, "vae", "model.safetensors"),
        os.path.join(vae_dir, "model.safetensors"),
    )

    image_encoder_dir = os.path.join(vae_dir, "ltx23_image_encoder")
    _ensure_dir(image_encoder_dir)
    _link_or_copy_file(
        os.path.join(config_donor_dir, "vae", "config.json"),
        os.path.join(image_encoder_dir, "config.json"),
    )
    _repack_ltx23_image_encoder_weights(
        os.path.join(config_donor_dir, "vae", "model.safetensors"),
        os.path.join(image_encoder_dir, "model.safetensors"),
    )

    _link_or_copy_file(
        os.path.join(source_dir, "ltx-2.3-22b-distilled-lora-384.safetensors"),
        os.path.join(output_dir, "ltx-2.3-22b-distilled-lora-384.safetensors"),
    )
    _link_or_copy_file(
        os.path.join(source_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        os.path.join(output_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    )
