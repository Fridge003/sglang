# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoArchConfig

WAN_S2V_SAMPLE_NEG_PROMPT = (
    "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass
class WanS2VArchConfig(WanVideoArchConfig):
    # Official Wan2.2 S2V transformer defaults.
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    hidden_size: int = 5120
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: bool = True
    eps: float = 1e-6

    # S2V-specific conditioning controls from the official implementation.
    cond_dim: int = 16
    audio_dim: int = 1024
    num_audio_token: int = 4
    enable_adain: bool = True
    adain_mode: str = "attn_norm"
    audio_inject_layers: list[int] = field(
        default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39]
    )
    zero_init: bool = True
    zero_timestep: bool = True
    enable_motioner: bool = False
    add_last_motion: bool = True
    enable_tsm: bool = False
    trainable_token_pos_emb: bool = False
    motion_token_num: int = 1024
    enable_framepack: bool = True
    framepack_drop_mode: str = "padd"
    model_type: str = "s2v"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class WanS2VConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=WanS2VArchConfig)
    prefix: str = "WanS2V"
    sample_neg_prompt: str = WAN_S2V_SAMPLE_NEG_PROMPT
