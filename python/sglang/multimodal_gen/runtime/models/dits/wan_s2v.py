# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import types
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.models.dits import WanS2VConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.wan_s2v_native_impl import WanModelS2V
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


_WAN_S2V_SAMPLE_NEG_PROMPT = (
    "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


class WanS2VTransformer3DModel(torch.nn.Module, OffloadableDiTMixin):
    _aliases = ["WanS2VTransformer3DModel"]
    _fsdp_shard_conditions = WanS2VConfig()._fsdp_shard_conditions
    _compile_conditions = WanS2VConfig()._compile_conditions
    _supported_attention_backends = WanS2VConfig()._supported_attention_backends
    param_names_mapping = WanS2VConfig().param_names_mapping
    reverse_param_names_mapping = WanS2VConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanS2VConfig().lora_param_names_mapping
    layer_names = ["blocks"]

    def __init__(
        self,
        noise_model: torch.nn.Module,
        *,
        component_model_path: str,
        config: dict[str, Any],
        config_obj: Any,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.noise_model = noise_model
        self.blocks = getattr(noise_model, "blocks", None)
        self.component_model_path = component_model_path
        self.config = config
        self.device_ = device
        self.param_dtype = config_obj.param_dtype
        self.num_train_timesteps = int(config_obj.num_train_timesteps)
        self.sample_neg_prompt = config_obj.sample_neg_prompt
        self.motion_frames = int(config_obj.transformer.motion_frames)
        self.drop_first_motion = bool(config_obj.drop_first_motion)
        self.fps = int(config_obj.sample_fps)
        self.audio_sample_m = 0
        self.supports_standard_denoising = True

    @property
    def device(self):
        return self.device_

    @staticmethod
    def _resolve_existing_path(
        component_model_path: str, path_value: str | None
    ) -> str:
        if not path_value:
            raise ValueError(
                "WanS2VTransformer3DModel config is missing a required path"
            )
        path_value = os.path.expanduser(path_value)
        if os.path.isabs(path_value):
            resolved = path_value
        else:
            resolved = os.path.join(component_model_path, path_value)
        if not os.path.exists(resolved):
            raise ValueError(f"Resolved path does not exist: {resolved}")
        return resolved

    @classmethod
    def _resolve_attention_backend(
        cls, server_args, config: dict[str, Any]
    ) -> AttentionBackendEnum:
        backend = getattr(server_args, "attention_backend", None)
        backend = str(config.get("attention_backend", "auto") if backend is None else backend).lower()

        # Keep the resolver aligned with WanTransformer3DModel: normalize the
        # public CLI/backend names first, then validate against the model's
        # supported backend set.
        backend_aliases = {
            "auto": AttentionBackendEnum.NO_ATTENTION,
            "fa": AttentionBackendEnum.FA,
            "fa2": AttentionBackendEnum.FA,
            "flash": AttentionBackendEnum.FA,
            "flashattention": AttentionBackendEnum.FA,
            "flash_attention": AttentionBackendEnum.FA,
            "torch_sdpa": AttentionBackendEnum.TORCH_SDPA,
            "sdpa": AttentionBackendEnum.TORCH_SDPA,
        }
        backend_enum = backend_aliases.get(backend)
        if backend_enum is None:
            raise ValueError(f"Unsupported Wan S2V attention backend: {backend}")
        if (
            backend_enum not in cls._supported_attention_backends
            and backend_enum is not AttentionBackendEnum.NO_ATTENTION
        ):
            raise ValueError(
                f"Wan S2V attention backend {backend_enum} is not supported"
            )
        return backend_enum

    @staticmethod
    def _build_runtime_config(config: dict[str, Any]) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            param_dtype=torch.bfloat16,
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            sample_neg_prompt=str(
                config.get("sample_neg_prompt", _WAN_S2V_SAMPLE_NEG_PROMPT)
            ),
            drop_first_motion=bool(config.get("drop_first_motion", True)),
            sample_fps=int(config.get("sample_fps", 16)),
            transformer=types.SimpleNamespace(
                motion_frames=int(config.get("motion_frames", 73))
            ),
        )

    @classmethod
    def from_component_path(
        cls,
        component_model_path: str,
        server_args,
        config: dict[str, Any],
    ):
        checkpoint_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_checkpoint_root", "checkpoints")
        )

        config = dict(config)
        attention_backend = cls._resolve_attention_backend(
            server_args, config
        )
        config["attention_backend"] = str(attention_backend)
        config_obj = cls._build_runtime_config(config)
        local_device = get_local_torch_device()
        logger.info(
            "Creating native WanModelS2V directly from %s",
            checkpoint_root,
        )
        noise_model = WanModelS2V.from_pretrained(
            checkpoint_root,
            torch_dtype=config_obj.param_dtype,
            device_map=local_device,
        )
        noise_model.eval().requires_grad_(False)

        if bool(config.get("convert_model_dtype", False)):
            noise_model.to(config_obj.param_dtype)
        if not bool(config.get("init_on_cpu", True)):
            noise_model.to(local_device)

        return cls(
            noise_model=noise_model,
            component_model_path=component_model_path,
            config=config,
            config_obj=config_obj,
            device=local_device,
        )

    def get_default_negative_prompt(self) -> str:
        return self.sample_neg_prompt

    def _normalize_infer_frames(self, num_frames: int) -> int:
        infer_frames = max(int(num_frames) - 1, 4)
        if infer_frames % 4 != 0:
            infer_frames = max((infer_frames // 4) * 4, 4)
        return infer_frames

    def get_size_less_than_area(
        self,
        height: int,
        width: int,
        *,
        target_area: int = 1024 * 704,
        divisor: int = 64,
    ) -> tuple[int, int]:
        if height * width <= target_area:
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            max_upper_area = target_area
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area
            min_scale = (-b + (b**2 - 2 * a * c) ** 0.5) / (2 * a)
            max_scale = (max_upper_area / (height * width)) ** 0.5

        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            padded_height = new_height + pad_height
            padded_width = new_width + pad_width
            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width

        aspect_ratio = width / height
        target_width = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
        target_height = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)
        if target_width >= width or target_height >= height:
            target_width = int(width // divisor * divisor)
            target_height = int(height // divisor * divisor)
        return target_height, target_width

    def get_generation_size(self, *, image_path: str) -> tuple[int, int]:
        ref_image = np.array(Image.open(image_path).convert("RGB"))
        height, width = ref_image.shape[:2]
        return self.get_size_less_than_area(
            int(height),
            int(width),
            target_area=int(self.config.get("max_area", 720 * 1280)),
        )

    def prepare_standard_s2v_latents(
        self,
        *,
        latent_shape: tuple[int, ...],
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        from diffusers.utils.torch_utils import randn_tensor

        return randn_tensor(
            latent_shape,
            generator=generator,
            device=self.device,
            dtype=self.param_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        ref_latents: torch.Tensor | None = None,
        motion_latents: torch.Tensor | None = None,
        cond_states: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        motion_frames: list[int] | tuple[int, int] | None = None,
        drop_motion_frames: bool = False,
        add_last_motion: bool | int | None = None,
        seq_len: int | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        del guidance
        if timestep is None and "t" in kwargs:
            timestep = kwargs.pop("t")
        if timestep is None:
            raise ValueError("Wan S2V forward requires timestep")
        if encoder_hidden_states is None:
            raise ValueError("Wan S2V forward requires encoder_hidden_states")
        if isinstance(encoder_hidden_states, list):
            if len(encoder_hidden_states) == 0:
                raise ValueError("encoder_hidden_states list cannot be empty")
            context = encoder_hidden_states
        elif isinstance(encoder_hidden_states, torch.Tensor):
            if encoder_hidden_states.ndim == 3:
                context = [
                    encoder_hidden_states[i]
                    for i in range(encoder_hidden_states.shape[0])
                ]
            else:
                context = [encoder_hidden_states]
        else:
            raise TypeError(
                "Wan S2V encoder_hidden_states must be a tensor or list of tensors"
            )
        if seq_len is None:
            seq_len = int(
                hidden_states.shape[2]
                * hidden_states.shape[3]
                * hidden_states.shape[4]
                // 4
            )
        if ref_latents is None or motion_latents is None or cond_states is None:
            raise ValueError(
                "Wan S2V forward requires ref_latents, motion_latents, and cond_states"
            )
        if motion_frames is None:
            motion_frames = [
                self.motion_frames,
                (self.motion_frames + 3) // 4,
            ]
        if add_last_motion is None:
            add_last_motion = 2
        output = self.noise_model(
            hidden_states,
            t=timestep,
            context=context,
            seq_len=seq_len,
            ref_latents=ref_latents,
            motion_latents=motion_latents,
            cond_states=cond_states,
            audio_input=audio_input,
            motion_frames=motion_frames,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
            **kwargs,
        )
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise ValueError(
                    f"Wan S2V noise model returned unexpected output length: {len(output)}"
                )
            return output[0]
        return output


EntryClass = WanS2VTransformer3DModel
