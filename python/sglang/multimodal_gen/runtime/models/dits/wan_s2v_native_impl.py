# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AdaLayerNorm
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n = x.size(2)
    output = []
    for i, _ in enumerate(x):
        seq_len = x.size(1)
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = freqs[i, :seq_len]
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).to(dtype=x.dtype)


@amp.autocast(enabled=False)
def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2
    if isinstance(freqs, list):
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64))
    seq_bucket = [0]
    if not isinstance(grid_sizes, list):
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not isinstance(g, list):
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]
            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)
                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1,
                    ).reshape(seq_len, 1, -1)
                else:
                    freqs_i = trainable_freqs.unsqueeze(1)
                output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


def _gather_debug_tensor(x, grid_sizes, num_replicated_suffix):
    if get_sp_world_size() == 1:
        return x
    while isinstance(grid_sizes, list):
        grid_sizes = grid_sizes[-1]
    local_video_len = x.shape[1] - num_replicated_suffix
    num_frames = int(grid_sizes[0, 0].item())
    local_tokens_per_frame = local_video_len // num_frames
    video = x[:, :local_video_len].view(
        x.shape[0], num_frames, local_tokens_per_frame, x.shape[2]
    )
    video = sequence_model_parallel_all_gather(video.contiguous(), dim=2)
    return torch.cat(
        [video.reshape(x.shape[0], -1, x.shape[2]), x[:, local_video_len:]], dim=1
    )


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return out.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        skip_sequence_parallel: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            skip_sequence_parallel=skip_sequence_parallel,
        )

    def forward(self, x, seq_lens, grid_sizes, freqs, num_replicated_suffix=0):
        del seq_lens
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        x = self.attn(
            rope_apply(q, grid_sizes, freqs),
            rope_apply(k, grid_sizes, freqs),
            v,
            num_replicated_suffix=num_replicated_suffix,
        )
        return self.o(x.flatten(2))


class WanCrossAttention(WanSelfAttention):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
            skip_sequence_parallel=True,
        )

    def forward(self, x, context, context_lens):
        del context_lens
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = self.attn(q, k, v)
        return self.o(x.flatten(2))


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        num_replicated_suffix=0,
    ):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
            num_replicated_suffix=num_replicated_suffix,
        )
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            return self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))


class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode='replicate', **kwargs):
        super().__init__()
        self.time_causal_padding = (kernel_size - 1, 0)
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, self.time_causal_padding, mode=self.pad_mode))


class MotionEncoderTC(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, need_global=True, dtype=None, device=None):
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x_ori = x.clone()
        b, _, _ = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, 'b (n c) t -> (b n) t c', n=self.num_heads)
        x = self.act(self.norm1(x))
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = self.act(self.norm2(rearrange(x, 'b c t -> b t c')))
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = self.act(self.norm3(rearrange(x, 'b c t -> b t c')))
        x = rearrange(x, '(b n) t c -> b t n c', b=b)
        x_local = torch.cat([x, self.padding_tokens.repeat(b, x.shape[1], 1, 1)], dim=-2)
        if not self.need_global:
            return x_local
        x = self.conv1_global(x_ori)
        x = self.act(self.norm1(rearrange(x, 'b c t -> b t c')))
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = self.act(self.norm2(rearrange(x, 'b c t -> b t c')))
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = self.act(self.norm3(rearrange(x, 'b c t -> b t c')))
        x = self.final_linear(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)
        return x, x_local


class CausalAudioEncoder(nn.Module):
    def __init__(self, dim=5120, num_layers=25, out_dim=2048, video_rate=8, num_token=4, need_global=False):
        super().__init__()
        self.encoder = MotionEncoderTC(in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global)
        self.weights = nn.Parameter(torch.ones((1, num_layers, 1, 1)) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features):
        with amp.autocast(dtype=torch.float32):
            weights = self.act(self.weights)
            weighted_feat = ((features * weights) / weights.sum(dim=1, keepdims=True)).sum(dim=1)
            weighted_feat = weighted_feat.permute(0, 2, 1)
            return self.encoder(weighted_feat)


class AudioCrossAttention(WanCrossAttention):
    pass


class AudioInjectorWAN(nn.Module):
    def __init__(self, all_modules, all_module_names, dim=2048, num_heads=32, inject_layer=(0, 27), enable_adain=False, adain_dim=2048, need_adain_ont=False):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_module_names, all_modules):
            if isinstance(mod, WanAttentionBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1
        self.injector = nn.ModuleList([AudioCrossAttention(dim=dim, num_heads=num_heads, qk_norm=True) for _ in range(audio_injector_id)])
        self.injector_pre_norm_feat = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(audio_injector_id)])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([AdaLayerNorm(output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1) for _ in range(audio_injector_id)])
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(audio_injector_id)])


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [parent_name if parent_name else 'root'], [model]
    for name, child in model.named_children():
        child_name = f'{parent_name}.{name}' if parent_name else name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class FramePackMotioner(nn.Module):
    def __init__(self, inner_dim=1024, num_heads=16, zip_frame_buckets=(1, 2, 16), drop_mode='drop'):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(zip_frame_buckets, dtype=torch.long)
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        d = inner_dim // num_heads
        self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        self.drop_mode = drop_mode

    def forward(self, motion_latents, add_last_motion=2):
        mot, mot_remb = [], []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height, lat_width, device=m.device, dtype=m.dtype)
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]
            if add_last_motion < 2 and self.drop_mode != 'drop':
                zero_end_frame = self.zip_frame_buckets[: self.zip_frame_buckets.__len__() - add_last_motion - 1].sum()
                padd_lat[:, -zero_end_frame:] = 0
            padd_lat = padd_lat.unsqueeze(0)
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -self.zip_frame_buckets.sum() :, :, :].split(list(self.zip_frame_buckets)[::-1], dim=2)
            clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)
            if add_last_motion < 2 and self.drop_mode == 'drop':
                clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x
            motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = [] if add_last_motion < 2 and self.drop_mode == 'drop' else [[torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1), torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1)]]
            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = [] if add_last_motion < 1 and self.drop_mode == 'drop' else [[torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1), torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1), torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1)]]
            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [[torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1), torch.tensor([end_time_id, lat_height // 8, lat_width // 8]).unsqueeze(0).repeat(1, 1), torch.tensor([self.zip_frame_buckets[2], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1)]]
            motion_rope_emb = rope_precompute(motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads, self.inner_dim // self.num_heads), grid_sizes + grid_sizes_2x + grid_sizes_4x, self.freqs, start=None)
            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


class HeadS2V(Head):
    def forward(self, x, e):
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class WanS2VSelfAttention(WanSelfAttention):
    pass


class WanS2VAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__(
            dim,
            ffn_dim,
            num_heads,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )
        self.self_attn = WanS2VSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        num_replicated_suffix=0,
    ):
        debug_dump_dir = os.getenv("SGLANG_WAN_S2V_INTERNAL_DUMP_DIR")
        debug_block_idx = int(os.getenv("SGLANG_WAN_S2V_INTERNAL_DUMP_BLOCK_IDX", "-1"))
        debug_enabled = (
            debug_dump_dir
            and getattr(self, "_debug_dump_enabled", False)
            and getattr(self, "_debug_block_idx", -1) == debug_block_idx
        )
        dump_rank = get_sp_group().rank_in_group if get_sp_world_size() > 1 else 0

        def dump_tensor(name, tensor):
            if not debug_enabled or dump_rank != 0:
                return
            tensor = _gather_debug_tensor(tensor, grid_sizes, num_replicated_suffix)
            os.makedirs(debug_dump_dir, exist_ok=True)
            torch.save(tensor.detach().cpu(), os.path.join(debug_dump_dir, name))

        seg_idx = min(max(0, e[1].item()), x.size(1))
        seg_idx = [0, seg_idx, x.size(1)]
        e = e[0]
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation.unsqueeze(2) + e).chunk(6, dim=1)
        e = [element.squeeze(1) for element in e]
        norm_x = self.norm1(x).float()
        norm_x = torch.cat([norm_x[:, seg_idx[i]:seg_idx[i+1]] * (1 + e[1][:, i:i+1]) + e[0][:, i:i+1] for i in range(2)], dim=1)
        y = self.self_attn(
            norm_x,
            seq_lens,
            grid_sizes,
            freqs,
            num_replicated_suffix=num_replicated_suffix,
        )
        dump_tensor("self_attn.pt", y)
        with amp.autocast(dtype=torch.float32):
            y = torch.cat([y[:, seg_idx[i]:seg_idx[i+1]] * e[2][:, i:i+1] for i in range(2)], dim=1)
            x = x + y
        dump_tensor("post_self_residual.pt", x)
        cross = self.cross_attn(self.norm3(x), context, context_lens)
        dump_tensor("cross_attn.pt", cross)
        x = x + cross
        norm2_x = self.norm2(x).float()
        norm2_x = torch.cat([norm2_x[:, seg_idx[i]:seg_idx[i+1]] * (1 + e[4][:, i:i+1]) + e[3][:, i:i+1] for i in range(2)], dim=1)
        y = self.ffn(norm2_x)
        dump_tensor("ffn.pt", y)
        with amp.autocast(dtype=torch.float32):
            y = torch.cat([y[:, seg_idx[i]:seg_idx[i+1]] * e[5][:, i:i+1] for i in range(2)], dim=1)
            x = x + y
        dump_tensor("block_out.pt", x)
        return x


class WanModelS2V(ModelMixin, ConfigMixin):
    ignore_for_config = ['args', 'kwargs', 'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size']
    _no_split_modules = ['WanS2VAttentionBlock']

    @register_to_config
    def __init__(self, cond_dim=0, audio_dim=5120, num_audio_token=4, enable_adain=False, adain_mode='attn_norm', audio_inject_layers=(0, 4, 8, 12, 16, 20, 24, 27), zero_init=False, zero_timestep=False, enable_motioner=False, add_last_motion=True, enable_tsm=False, trainable_token_pos_emb=False, motion_token_num=1024, enable_framepack=True, framepack_drop_mode='padd', model_type='s2v', patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096, out_dim=16, num_heads=16, num_layers=32, window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6, *args, **kwargs):
        super().__init__()
        assert model_type == 's2v'
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self._supported_attention_backends = {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
        }
        self.blocks = nn.ModuleList([
            WanS2VAttentionBlock(
                dim,
                ffn_dim,
                num_heads,
                window_size,
                qk_norm,
                cross_attn_norm,
                eps,
                supported_attention_backends=self._supported_attention_backends,
            )
            for _ in range(num_layers)
        ])
        for idx, block in enumerate(self.blocks):
            block._debug_block_idx = idx
            block._debug_dump_enabled = False
        self.head = HeadS2V(dim, out_dim, patch_size, eps)
        d = dim // num_heads
        self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        self.init_weights()
        self.sp_size = get_sp_world_size()
        self.use_context_parallel = self.sp_size > 1
        if cond_dim > 0:
            self.cond_encoder = nn.Conv3d(cond_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.enbale_adain = enable_adain
        self.casual_audio_encoder = CausalAudioEncoder(dim=audio_dim, out_dim=self.dim, num_token=num_audio_token, need_global=enable_adain)
        all_modules, all_module_names = torch_dfs(self.blocks, parent_name='root.transformer_blocks')
        self.audio_injector = AudioInjectorWAN(all_modules, all_module_names, dim=self.dim, num_heads=self.num_heads, inject_layer=list(audio_inject_layers), enable_adain=enable_adain, adain_dim=self.dim, need_adain_ont=adain_mode != 'attn_norm')
        self.adain_mode = adain_mode
        self.trainable_cond_mask = nn.Embedding(3, self.dim)
        if zero_init:
            self.zero_init_weights()
        self.zero_timestep = zero_timestep
        self.enable_motioner = enable_motioner
        self.add_last_motion = add_last_motion
        self.enable_framepack = enable_framepack
        if enable_motioner:
            raise NotImplementedError('Native Wan S2V motioner path is not implemented')
        if enable_framepack:
            self.frame_packer = FramePackMotioner(inner_dim=self.dim, num_heads=self.num_heads, zip_frame_buckets=[1, 2, 16], drop_mode=framepack_drop_mode)
        self._debug_block_dump_dir = os.getenv("SGLANG_WAN_S2V_BLOCK_DUMP_DIR")
        self._debug_block_dump_call_idx = int(
            os.getenv("SGLANG_WAN_S2V_BLOCK_DUMP_CALL_IDX", "0")
        )
        block_indexes = os.getenv("SGLANG_WAN_S2V_BLOCK_DUMP_INDEXES")
        self._debug_block_dump_indexes = (
            {int(index) for index in block_indexes.split(",")}
            if block_indexes
            else None
        )
        self._forward_call_idx = 0

    def zero_init_weights(self):
        with torch.no_grad():
            self.trainable_cond_mask = zero_module(self.trainable_cond_mask)
            if hasattr(self, 'cond_encoder'):
                self.cond_encoder = zero_module(self.cond_encoder)
            for i in range(len(self.audio_injector.injector)):
                self.audio_injector.injector[i].o = zero_module(self.audio_injector.injector[i].o)
                if self.enbale_adain:
                    self.audio_injector.injector_adain_layers[i].linear = zero_module(self.audio_injector.injector_adain_layers[i].linear)

    def process_motion_frame_pack(self, motion_latents, drop_motion_frames=False, add_last_motion=2):
        flattern_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot], [m[:, :0] for m in mot_remb]
        return flattern_mot, mot_remb

    def inject_motion(self, x, seq_lens, rope_embs, mask_input, motion_latents, drop_motion_frames=False, add_last_motion=True):
        mot, mot_remb = self.process_motion_frame_pack(motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=add_last_motion)
        if len(mot) > 0:
            x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + torch.tensor([r.size(1) for r in mot], dtype=torch.long)
            rope_embs = [torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)]
            mask_input = [torch.cat([m, 2 * torch.ones([1, u.shape[1] - m.shape[1]], device=m.device, dtype=m.dtype)], dim=1) for m, u in zip(mask_input, x)]
        return x, seq_lens, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states):
        if block_idx in self.audio_injector.injected_block_id:
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb
            num_frames = audio_emb.shape[1]
            input_hidden_states = hidden_states[:, : self.local_original_seq_len].clone()
            input_hidden_states = rearrange(input_hidden_states, 'b (t n) c -> (b t) n c', t=num_frames)
            if self.enbale_adain and self.adain_mode == 'attn_norm':
                audio_emb_global = rearrange(self.audio_emb_global, 'b t n c -> (b t) n c')
                attn_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](input_hidden_states, temb=audio_emb_global[:, 0])
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
            attn_audio_emb = rearrange(audio_emb, 'b t n c -> (b t) n c', t=num_frames)
            residual_out = self.audio_injector.injector[audio_attn_id](x=attn_hidden_states, context=attn_audio_emb, context_lens=torch.ones(attn_hidden_states.shape[0], dtype=torch.long, device=attn_hidden_states.device) * attn_audio_emb.shape[1])
            residual_out = rearrange(residual_out, '(b t) n c -> b (t n) c', t=num_frames)
            hidden_states[:, : self.local_original_seq_len] = hidden_states[:, : self.local_original_seq_len] + residual_out
        return hidden_states

    def forward(self, x, t, context, seq_len, ref_latents, motion_latents, cond_states, audio_input=None, motion_frames=(17, 5), add_last_motion=2, drop_motion_frames=False, *extra_args, **extra_kwargs):
        forward_batch = get_forward_context().forward_batch
        sequence_shard_enabled = (
            forward_batch is not None
            and forward_batch.enable_sequence_shard
            and self.sp_size > 1
        )
        add_last_motion = self.add_last_motion * add_last_motion
        audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input], dim=-1)
        audio_emb_res = self.casual_audio_encoder(audio_input)
        if self.enbale_adain:
            audio_emb_global, audio_emb = audio_emb_res
            self.audio_emb_global = audio_emb_global[:, motion_frames[1]:].clone()
        else:
            audio_emb = audio_emb_res
        self.merged_audio_emb = audio_emb[:, motion_frames[1]:, :]
        num_frames = self.merged_audio_emb.shape[1]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
        x = [x_ + pose for x_, pose in zip(x, cond)]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]
        self.lat_motion_frames = motion_latents[0].shape[1]
        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref_grid_sizes = [[torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1), torch.tensor([31, height, width]).unsqueeze(0).repeat(batch_size, 1), torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1)]]
        ref = [r.flatten(2).transpose(1, 2) for r in ref]
        self.original_seq_len = seq_lens[0]
        seq_lens = seq_lens + torch.tensor([r.size(1) for r in ref], dtype=torch.long)
        grid_sizes = grid_sizes + ref_grid_sizes
        x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]
        mask_input = [torch.zeros([1, u.shape[1]], dtype=torch.long, device=x[0].device) for u in x]
        for i in range(len(mask_input)):
            mask_input[i][:, self.original_seq_len:] = 1
        x = torch.cat(x)
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.dim // self.num_heads
        self.pre_compute_freqs = rope_precompute(x.detach().view(b, s, n, d), grid_sizes, self.freqs, start=None)
        x = [u.unsqueeze(0) for u in x]
        self.pre_compute_freqs = [u.unsqueeze(0) for u in self.pre_compute_freqs]
        x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(x, seq_lens, self.pre_compute_freqs, mask_input, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=add_last_motion)
        x = torch.cat(x, dim=0)
        self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)
        condition_suffix_len = x.shape[1] - self.original_seq_len
        self.local_original_seq_len = self.original_seq_len
        seq_len_full = x.shape[1]
        seq_shard_pad = 0
        if sequence_shard_enabled:
            if self.original_seq_len % num_frames != 0:
                raise ValueError(
                    f"Wan S2V video sequence length {self.original_seq_len} must be divisible by num_frames {num_frames}"
                )
            tokens_per_frame = self.original_seq_len // num_frames
            if tokens_per_frame % self.sp_size != 0:
                raise ValueError(
                    f"Wan S2V tokens per frame {tokens_per_frame} must be divisible by sp_size {self.sp_size}"
                )
            sp_rank = get_sp_group().rank_in_group
            local_tokens_per_frame = tokens_per_frame // self.sp_size
            self.local_original_seq_len = num_frames * local_tokens_per_frame
            frame_slice = slice(
                sp_rank * local_tokens_per_frame,
                (sp_rank + 1) * local_tokens_per_frame,
            )
            x_video = x[:, : self.original_seq_len].view(
                x.shape[0], num_frames, tokens_per_frame, x.shape[2]
            )[:, :, frame_slice]
            x = torch.cat([x_video.reshape(x.shape[0], -1, x.shape[2]), x[:, self.original_seq_len :]], dim=1)
            freqs_video = self.pre_compute_freqs[:, : self.original_seq_len].view(
                self.pre_compute_freqs.shape[0],
                num_frames,
                tokens_per_frame,
                self.pre_compute_freqs.shape[2],
                self.pre_compute_freqs.shape[3],
            )[:, :, frame_slice]
            self.pre_compute_freqs = torch.cat(
                [
                    freqs_video.reshape(
                        self.pre_compute_freqs.shape[0],
                        -1,
                        self.pre_compute_freqs.shape[2],
                        self.pre_compute_freqs.shape[3],
                    ),
                    self.pre_compute_freqs[:, self.original_seq_len :],
                ],
                dim=1,
            )
            mask_video = mask_input[:, : self.original_seq_len].view(
                mask_input.shape[0], num_frames, tokens_per_frame
            )[:, :, frame_slice]
            mask_input = torch.cat(
                [mask_video.reshape(mask_input.shape[0], -1), mask_input[:, self.original_seq_len :]],
                dim=1,
            )
            seq_lens = torch.full_like(
                seq_lens, self.local_original_seq_len + condition_suffix_len
            )
        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)
        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        if self.zero_timestep:
            e = e[:-1]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            e0 = torch.cat([e0.unsqueeze(2), zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1)], dim=2)
            e0 = [e0, self.local_original_seq_len]
        else:
            e0 = [e0.unsqueeze(2).repeat(1, 1, 2, 1), 0]
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.pre_compute_freqs,
            context=context,
            context_lens=None,
            num_replicated_suffix=condition_suffix_len if sequence_shard_enabled else 0,
        )
        dump_this_call = (
            self._debug_block_dump_dir is not None
            and self._forward_call_idx == self._debug_block_dump_call_idx
        )
        for block in self.blocks:
            block._debug_dump_enabled = dump_this_call
        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            x = self.after_transformer_block(idx, x)
            if dump_this_call and (
                self._debug_block_dump_indexes is None
                or idx in self._debug_block_dump_indexes
            ):
                dump_x = x
                if sequence_shard_enabled:
                    dump_video = dump_x[:, : self.local_original_seq_len].view(
                        dump_x.shape[0],
                        num_frames,
                        self.local_original_seq_len // num_frames,
                        dump_x.shape[2],
                    )
                    dump_video = sequence_model_parallel_all_gather(
                        dump_video.contiguous(), dim=2
                    )
                    dump_x = torch.cat(
                        [
                            dump_video.reshape(dump_x.shape[0], -1, dump_x.shape[2]),
                            dump_x[:, self.local_original_seq_len :],
                        ],
                        dim=1,
                    )
                if not sequence_shard_enabled or get_sp_group().rank_in_group == 0:
                    os.makedirs(self._debug_block_dump_dir, exist_ok=True)
                    torch.save(
                        dump_x.detach().cpu(),
                        os.path.join(
                            self._debug_block_dump_dir,
                            f"call_{self._forward_call_idx:02d}_block_{idx:02d}.pt",
                        ),
                    )
        if sequence_shard_enabled:
            x_video = x[:, : self.local_original_seq_len].view(
                x.shape[0], num_frames, self.local_original_seq_len // num_frames, x.shape[2]
            )
            x_video = sequence_model_parallel_all_gather(x_video.contiguous(), dim=2)
            x = torch.cat([x_video.reshape(x.shape[0], -1, x.shape[2]), x[:, self.local_original_seq_len :]], dim=1)
            if seq_shard_pad > 0:
                x = x[:, :seq_len_full]
        x = x[:, : self.original_seq_len]
        x = self.head(x, e)
        self._forward_call_idx += 1
        x = self.unpatchify(x, original_grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, self.out_dim)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            out.append(u.reshape(self.out_dim, *[i * j for i, j in zip(v, self.patch_size)]))
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)
