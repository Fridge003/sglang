# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WanS2VTransformer3DModel(torch.nn.Module, OffloadableDiTMixin):
    _aliases = ["WanS2VTransformer3DModel"]
    _sdpa_warned_padding_mask = False
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

    @staticmethod
    def _sdpa_flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        del window_size, deterministic, version
        if q_scale is not None:
            q = q * q_scale
        if softmax_scale is not None:
            q = q * softmax_scale

        if q_lens is not None or k_lens is not None:
            if not WanS2VTransformer3DModel._sdpa_warned_padding_mask:
                logger.warning(
                    "Wan S2V SDPA fallback ignores q_lens/k_lens padding masks; "
                    "this is intended only for compatibility validation."
                )
                WanS2VTransformer3DModel._sdpa_warned_padding_mask = True

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
        )
        return out.transpose(1, 2).contiguous()

    @classmethod
    def _compatible_flash_attention(
        cls,
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        del dropout_p
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        assert q.device.type == "cuda" and q.size(-1) <= 256

        b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        if q_lens is None:
            q = half(q.flatten(0, 1))
            q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
        else:
            q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

        if k_lens is None:
            k = half(k.flatten(0, 1))
            v = half(v.flatten(0, 1))
            k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
        else:
            k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
            v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

        q = q.to(v.dtype)
        k = k.to(v.dtype)
        if q_scale is not None:
            q = q * q_scale

        module_attention = importlib.import_module("wan.modules.attention")
        fa3_func = getattr(
            importlib.import_module("flash_attn_interface"),
            "flash_attn_varlen_func",
            None,
        )
        fa2_func = getattr(
            importlib.import_module("flash_attn"), "flash_attn_varlen_func", None
        )

        use_fa3 = (
            (version is None or version == 3)
            and getattr(module_attention, "FLASH_ATTN_3_AVAILABLE", False)
            and fa3_func is not None
        )
        if use_fa3:
            x = fa3_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
                window_size=window_size,
            )
            if isinstance(x, tuple):
                x = x[0]
            return x.unflatten(0, (b, lq)).type(out_dtype)

        if fa2_func is None:
            raise RuntimeError(
                "No compatible flash attention varlen kernel is available for Wan S2V"
            )
        x = fa2_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        return x.unflatten(0, (b, lq)).type(out_dtype)

    @classmethod
    def _patch_attention_backend(cls, config: dict[str, Any]):
        backend = str(config.get("attention_backend", "auto")).lower()
        if backend not in {"auto", "flash", "sdpa"}:
            raise ValueError(f"Unsupported Wan attention backend: {backend}")

        module_attention = importlib.import_module("wan.modules.attention")
        if backend == "flash":
            module_attention.flash_attention = cls._compatible_flash_attention
            logger.info("Using official Wan flash attention backend (compat)")
            module_model = importlib.import_module("wan.modules.model")
            module_model.flash_attention = cls._compatible_flash_attention
            for module_name in (
                "wan.modules.s2v.model_s2v",
                "wan.modules.s2v.motioner",
                "wan.distributed.ulysses",
            ):
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    continue
                setattr(module, "flash_attention", cls._compatible_flash_attention)
            return

        if backend == "auto":
            has_fa2 = getattr(
                importlib.import_module("flash_attn"),
                "flash_attn_varlen_func",
                None,
            )
            has_fa3 = getattr(module_attention, "FLASH_ATTN_3_AVAILABLE", False)
            if has_fa2 or not has_fa3:
                module_attention.flash_attention = cls._compatible_flash_attention
                module_model = importlib.import_module("wan.modules.model")
                module_model.flash_attention = cls._compatible_flash_attention
                for module_name in (
                    "wan.modules.s2v.model_s2v",
                    "wan.modules.s2v.motioner",
                    "wan.distributed.ulysses",
                ):
                    try:
                        module = importlib.import_module(module_name)
                    except ModuleNotFoundError:
                        continue
                    setattr(module, "flash_attention", cls._compatible_flash_attention)
                logger.info("Using official Wan flash attention backend (auto compat)")
                return
            backend = "sdpa"

        logger.info("Patching official Wan attention backend to SDPA fallback")
        module_attention.flash_attention = cls._sdpa_flash_attention
        module_attention.FLASH_ATTN_2_AVAILABLE = False
        module_attention.FLASH_ATTN_3_AVAILABLE = False

        module_model = importlib.import_module("wan.modules.model")
        module_model.flash_attention = cls._sdpa_flash_attention
        for module_name in (
            "wan.modules.s2v.model_s2v",
            "wan.modules.s2v.motioner",
            "wan.distributed.ulysses",
        ):
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            setattr(module, "flash_attention", cls._sdpa_flash_attention)

    @classmethod
    def _resolve_attention_backend(cls, server_args, config: dict[str, Any]) -> str:
        backend = getattr(server_args, "attention_backend", None)
        if backend is None:
            return str(config.get("attention_backend", "auto")).lower()
        backend = str(backend).lower()
        if backend in {"torch_sdpa", "sdpa"}:
            return "sdpa"
        if backend in {"fa", "flash", "flashattention", "flash_attention"}:
            return "flash"
        return backend

    @classmethod
    def from_component_path(
        cls,
        component_model_path: str,
        server_args,
        config: dict[str, Any],
    ):
        code_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_code_root", "official_code")
        )
        checkpoint_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_checkpoint_root", "checkpoints")
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        task_name = str(config.get("wan_task_name", "s2v-14B"))
        module_configs = importlib.import_module("wan.configs")
        module_s2v = importlib.import_module("wan.speech2video")
        config = dict(config)
        config["attention_backend"] = cls._resolve_attention_backend(
            server_args, config
        )
        cls._patch_attention_backend(config)

        wan_configs = getattr(module_configs, "WAN_CONFIGS", None)
        if not isinstance(wan_configs, dict) or task_name not in wan_configs:
            raise ValueError(
                f"Official Wan config {task_name!r} not found under {code_root}"
            )

        config_obj = wan_configs[task_name]
        local_device = get_local_torch_device()
        use_sp = (
            max(
                getattr(server_args, "sp_degree", 1) or 1,
                getattr(server_args, "ulysses_degree", 1) or 1,
            )
            > 1
        )
        logger.info(
            "Creating WanModel_S2V directly from %s",
            checkpoint_root,
        )
        noise_model = module_s2v.WanModel_S2V.from_pretrained(
            checkpoint_root,
            torch_dtype=config_obj.param_dtype,
            device_map=local_device,
        )
        noise_model.eval().requires_grad_(False)
        if use_sp:
            for block in noise_model.blocks:
                block.self_attn.forward = types.MethodType(
                    module_s2v.sp_attn_forward_s2v, block.self_attn
                )
            noise_model.use_context_parallel = True

        if bool(config.get("convert_model_dtype", False)):
            noise_model.to(config_obj.param_dtype)
        if not bool(config.get("init_on_cpu", True)) or use_sp:
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
