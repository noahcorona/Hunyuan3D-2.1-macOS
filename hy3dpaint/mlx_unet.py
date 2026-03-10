"""MLX implementation of the Hunyuan3D inner UNet (UNet2DConditionModel).

Replaces the PyTorch inner UNet for generation passes (steps 2-15).
Runs entirely on Metal via MLX, achieving ~5x faster SDPA vs PyTorch MPS.

Architecture: standard diffusers UNet2DConditionModel with 16 custom
Basic2p5DTransformerBlocks replacing BasicTransformerBlock.

Layout: NHWC (MLX native). Conv2d weights need OIHW → OHWI transpose from PyTorch.
"""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    return x.reshape(B, H * scale, W * scale, C)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings (matching diffusers)."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half) / half)
    args = timesteps[:, None].astype(mx.float32) * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# 3D Rotary Position Embeddings
# ──────────────────────────────────────────────────────────────────────────────

def get_1d_rotary_pos_embed(dim: int, pos: mx.array, theta: float = 10000.0):
    """1D rotary embeddings → (cos, sin) each [N, dim]."""
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32)[: dim // 2] / dim))
    freqs = pos[:, None].astype(mx.float32) * freqs[None]
    # repeat_interleave(2, dim=1)
    cos = mx.repeat(mx.cos(freqs), 2, axis=1)
    sin = mx.repeat(mx.sin(freqs), 2, axis=1)
    return cos, sin


def get_3d_rotary_pos_embed(position: mx.array, embed_dim: int, voxel_resolution: int, theta: int = 10000):
    """3D RoPE from voxel indices. position: [..., 3], returns (cos, sin) each [..., embed_dim]."""
    dim_xy = embed_dim // 8 * 3
    dim_z = embed_dim // 8 * 2

    grid = mx.arange(voxel_resolution).astype(mx.float32)
    xy_cos, xy_sin = get_1d_rotary_pos_embed(dim_xy, grid, theta)
    z_cos, z_sin = get_1d_rotary_pos_embed(dim_z, grid, theta)

    flat = position.reshape(-1, 3).astype(mx.int32)
    x_cos, x_sin = xy_cos[flat[:, 0]], xy_sin[flat[:, 0]]
    y_cos, y_sin = xy_cos[flat[:, 1]], xy_sin[flat[:, 1]]
    zc, zs = z_cos[flat[:, 2]], z_sin[flat[:, 2]]

    cos = mx.concatenate([x_cos, y_cos, zc], axis=-1).reshape(*position.shape[:-1], embed_dim)
    sin = mx.concatenate([x_sin, y_sin, zs], axis=-1).reshape(*position.shape[:-1], embed_dim)
    return cos, sin


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array):
    """Apply RoPE. x: [B, heads, seq, dim], cos/sin: [B*?, seq, dim] → unsqueeze head dim."""
    cos = mx.expand_dims(cos, 1)  # [B, 1, seq, dim]
    sin = mx.expand_dims(sin, 1)

    # Rotate: [-x_imag, x_real] interleaved
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(x.shape)

    return (x.astype(mx.float32) * cos + x_rotated.astype(mx.float32) * sin).astype(x.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Core Modules
# ──────────────────────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, out_ch)
        self.linear_2 = nn.Linear(out_ch, out_ch)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))


class ResnetBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, groups: int = 32):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(groups, in_ch, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def __call__(self, x, temb):
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_emb_proj(nn.silu(temb))[:, None, None, :]
        h = nn.silu(self.norm2(h))
        h = self.conv2(h)
        if self.in_ch != self.out_ch:
            x = self.conv_shortcut(x)
        return x + h


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x):
        return self.conv(upsample_nearest(x))


# ──────────────────────────────────────────────────────────────────────────────
# Attention (SDPA)
# ──────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head attention with separate Q/K/V linear projections."""

    def __init__(self, dim: int, heads: int, cross_dim: Optional[int] = None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        kv_dim = cross_dim if cross_dim is not None else dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(kv_dim, dim, bias=False)
        self.to_v = nn.Linear(kv_dim, dim, bias=False)
        self.to_out_linear = nn.Linear(dim, dim)

    def __call__(self, x, context=None, rope=None):
        context = context if context is not None else x
        B, N, _ = x.shape
        q = self.to_q(x).reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        M = context.shape[1]
        k = self.to_k(context).reshape(B, M, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, M, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        if rope is not None:
            cos, sin = rope
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out_linear(out)


# ──────────────────────────────────────────────────────────────────────────────
# Basic2p5DTransformerBlock (MLX)
# ──────────────────────────────────────────────────────────────────────────────

class Basic2p5DTransformerBlock(nn.Module):
    """Custom transformer block with MDA, ref attn, multiview attn, DINO attn.

    All attention uses MLX SDPA (mx.fast.scaled_dot_product_attention).
    """

    def __init__(self, dim: int, heads: int, cross_dim: int = 1024,
                 ff_dim: Optional[int] = None, layer_name: str = ""):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.layer_name = layer_name

        # Norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # Self-attention (albedo)
        self.attn1_to_q = nn.Linear(dim, dim, bias=False)
        self.attn1_to_k = nn.Linear(dim, dim, bias=False)
        self.attn1_to_v = nn.Linear(dim, dim, bias=False)
        self.attn1_to_out = nn.Linear(dim, dim)
        # Self-attention (MR material)
        self.attn1_to_q_mr = nn.Linear(dim, dim, bias=False)
        self.attn1_to_k_mr = nn.Linear(dim, dim, bias=False)
        self.attn1_to_v_mr = nn.Linear(dim, dim, bias=False)
        self.attn1_to_out_mr = nn.Linear(dim, dim)

        # Reference attention (shared Q/K, separate V per PBR)
        self.attn_ref_to_q = nn.Linear(dim, dim, bias=False)
        self.attn_ref_to_k = nn.Linear(dim, dim, bias=False)
        self.attn_ref_to_v = nn.Linear(dim, dim, bias=False)  # albedo V
        self.attn_ref_to_out = nn.Linear(dim, dim)  # albedo out
        self.attn_ref_to_v_mr = nn.Linear(dim, dim, bias=False)  # MR V
        self.attn_ref_to_out_mr = nn.Linear(dim, dim)  # MR out

        # Multiview attention with RoPE
        self.attn_mv_to_q = nn.Linear(dim, dim, bias=False)
        self.attn_mv_to_k = nn.Linear(dim, dim, bias=False)
        self.attn_mv_to_v = nn.Linear(dim, dim, bias=False)
        self.attn_mv_to_out = nn.Linear(dim, dim)

        # Text cross-attention
        self.attn2_to_q = nn.Linear(dim, dim, bias=False)
        self.attn2_to_k = nn.Linear(cross_dim, dim, bias=False)
        self.attn2_to_v = nn.Linear(cross_dim, dim, bias=False)
        self.attn2_to_out = nn.Linear(dim, dim)

        # DINO cross-attention
        self.attn_dino_to_q = nn.Linear(dim, dim, bias=False)
        self.attn_dino_to_k = nn.Linear(cross_dim, dim, bias=False)
        self.attn_dino_to_v = nn.Linear(cross_dim, dim, bias=False)
        self.attn_dino_to_out = nn.Linear(dim, dim)

        # Feed-forward (GEGLU)
        ff_inner = ff_dim if ff_dim is not None else 4 * dim
        self.ff_proj = nn.Linear(dim, ff_inner * 2)  # GEGLU: gate + value
        self.ff_out = nn.Linear(ff_inner, dim)

    def _sdpa(self, q, k, v, B, rope=None):
        """Reshape to multi-head and run SDPA."""
        N_q, N_kv = q.shape[1], k.shape[1]
        q = q.reshape(B, N_q, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N_kv, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N_kv, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        if rope is not None:
            cos, sin = rope
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return out.transpose(0, 2, 1, 3).reshape(B, N_q, -1)

    def __call__(
        self,
        hidden_states: mx.array,         # (B*N_pbr*N_gen, seq, dim)
        encoder_hidden_states: mx.array,  # (B*N_pbr*N_gen, 77, 1024)
        num_in_batch: int,                # N_gen = 6
        condition_embed: mx.array,        # (B, N_gen*seq, dim)
        dino_hidden_states: mx.array,     # (B, 24, 1024)
        position_voxel_indices: Optional[dict] = None,
        ref_scale: float = 1.0,
        mva_scale: float = 1.0,
    ) -> mx.array:
        N_pbr = 2  # albedo + mr
        batch_size = hidden_states.shape[0]
        B = batch_size // (N_pbr * num_in_batch)

        # ── 0. Norm ──
        norm_hs = self.norm1(hidden_states)

        # ── 1. Material-Dimension Attention (MDA) ──
        # Reshape: (B*N_pbr*N, L, C) → (B, N_pbr, N, L, C)
        mda_hs = norm_hs.reshape(B, N_pbr, num_in_batch, -1, self.dim)

        # Albedo (pbr=0)
        albedo_hs = mda_hs[:, 0].reshape(B * num_in_batch, -1, self.dim)
        q_a = self.attn1_to_q(albedo_hs)
        k_a = self.attn1_to_k(albedo_hs)
        v_a = self.attn1_to_v(albedo_hs)
        attn_albedo = self._sdpa(q_a, k_a, v_a, B * num_in_batch)
        attn_albedo = self.attn1_to_out(attn_albedo)

        # MR (pbr=1)
        mr_hs = mda_hs[:, 1].reshape(B * num_in_batch, -1, self.dim)
        q_m = self.attn1_to_q_mr(mr_hs)
        k_m = self.attn1_to_k_mr(mr_hs)
        v_m = self.attn1_to_v_mr(mr_hs)
        attn_mr = self._sdpa(q_m, k_m, v_m, B * num_in_batch)
        attn_mr = self.attn1_to_out_mr(attn_mr)

        # Recombine: (B, N, L, C) → stack → (B, N_pbr, N, L, C) → (B*N_pbr*N, L, C)
        attn_albedo = attn_albedo.reshape(B, 1, num_in_batch, -1, self.dim)
        attn_mr = attn_mr.reshape(B, 1, num_in_batch, -1, self.dim)
        attn_output = mx.concatenate([attn_albedo, attn_mr], axis=1)
        attn_output = attn_output.reshape(batch_size, -1, self.dim)

        hidden_states = attn_output + hidden_states

        # ── 2. Reference Attention ──
        # Only albedo features for query (from norm_hs)
        # norm_hs: (B*N_pbr*N, L, C) → (B, N_pbr, N*L, C), take albedo (idx 0)
        ref_q_hs = norm_hs.reshape(B, N_pbr, num_in_batch, -1, self.dim)
        ref_q_hs = ref_q_hs[:, 0].reshape(B, -1, self.dim)  # (B, N*L, C)

        q_ref = self.attn_ref_to_q(ref_q_hs)
        k_ref = self.attn_ref_to_k(condition_embed)

        # Albedo V + out
        v_ref_a = self.attn_ref_to_v(condition_embed)
        ref_out_a = self._sdpa(q_ref, k_ref, v_ref_a, B)
        ref_out_a = self.attn_ref_to_out(ref_out_a)

        # MR V + out
        v_ref_m = self.attn_ref_to_v_mr(condition_embed)
        ref_out_m = self._sdpa(q_ref, k_ref, v_ref_m, B)
        ref_out_m = self.attn_ref_to_out_mr(ref_out_m)

        # Stack PBR and reshape back: (B, N_pbr, N, L, C) → (B*N_pbr*N, L, C)
        ref_out_a = ref_out_a.reshape(B, 1, num_in_batch, -1, self.dim)
        ref_out_m = ref_out_m.reshape(B, 1, num_in_batch, -1, self.dim)
        ref_out = mx.concatenate([ref_out_a, ref_out_m], axis=1)
        ref_out = ref_out.reshape(batch_size, -1, self.dim)

        # ref_scale may be float or (B,) array — broadcast to (B*N_pbr*N, 1, 1)
        if isinstance(ref_scale, mx.array) and ref_scale.ndim >= 1:
            rs = mx.repeat(mx.expand_dims(ref_scale, 1), num_in_batch * N_pbr, axis=1)
            rs = rs.reshape(-1, 1, 1)  # (B*N_pbr*N, 1, 1)
            hidden_states = rs * ref_out + hidden_states
        else:
            hidden_states = ref_scale * ref_out + hidden_states

        # ── 3. Multiview Attention (with RoPE) ──
        # Reshape: (B*N_pbr*N, L, C) → (B*N_pbr, N*L, C)
        mv_hs = norm_hs.reshape(B * N_pbr, num_in_batch * norm_hs.shape[1], self.dim)

        # Compute RoPE if position indices available
        rope = None
        seq_len = mv_hs.shape[1]
        if position_voxel_indices is not None and seq_len in position_voxel_indices:
            entry = position_voxel_indices[seq_len]
            voxel_idx = entry["voxel_indices"]  # (B, seq, 3)
            voxel_res = entry["voxel_resolution"]
            # Repeat for N_pbr
            voxel_idx_rep = mx.repeat(mx.expand_dims(voxel_idx, 1), N_pbr, axis=1)
            voxel_idx_rep = voxel_idx_rep.reshape(B * N_pbr, -1, 3)
            rope = get_3d_rotary_pos_embed(voxel_idx_rep, self.head_dim, voxel_res)

        q_mv = self.attn_mv_to_q(mv_hs)
        k_mv = self.attn_mv_to_k(mv_hs)
        v_mv = self.attn_mv_to_v(mv_hs)
        mv_out = self._sdpa(q_mv, k_mv, v_mv, B * N_pbr, rope=rope)
        mv_out = self.attn_mv_to_out(mv_out)

        # Reshape back: (B*N_pbr, N*L, C) → (B*N_pbr*N, L, C)
        mv_out = mv_out.reshape(batch_size, -1, self.dim)

        hidden_states = mva_scale * mv_out + hidden_states

        # ── 4. Text Cross-Attention ──
        norm_hs2 = self.norm2(hidden_states)
        q_t = self.attn2_to_q(norm_hs2)
        k_t = self.attn2_to_k(encoder_hidden_states)
        v_t = self.attn2_to_v(encoder_hidden_states)
        text_out = self._sdpa(q_t, k_t, v_t, batch_size)
        text_out = self.attn2_to_out(text_out)
        hidden_states = text_out + hidden_states

        # ── 5. DINO Cross-Attention ──
        # dino: (B, 24, 1024) → repeat for N_pbr*N_gen → (B*N_pbr*N, 24, 1024)
        dino_expanded = mx.repeat(mx.expand_dims(dino_hidden_states, 1), N_pbr * num_in_batch, axis=1)
        dino_expanded = dino_expanded.reshape(batch_size, -1, dino_hidden_states.shape[-1])
        q_d = self.attn_dino_to_q(norm_hs2)
        k_d = self.attn_dino_to_k(dino_expanded)
        v_d = self.attn_dino_to_v(dino_expanded)
        dino_out = self._sdpa(q_d, k_d, v_d, batch_size)
        dino_out = self.attn_dino_to_out(dino_out)
        hidden_states = dino_out + hidden_states

        # ── 6. Feed-Forward (GEGLU) ──
        # PyTorch GEGLU: value, gate = chunk(2); output = value * gelu(gate)
        # First half is value (passed through), second half is gate (GELU applied)
        norm_hs3 = self.norm3(hidden_states)
        ff = self.ff_proj(norm_hs3)
        ff_val, ff_gate = mx.split(ff, 2, axis=-1)
        ff = ff_val * nn.gelu(ff_gate)
        ff = self.ff_out(ff)
        hidden_states = ff + hidden_states

        return hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Transformer2D (wraps transformer blocks with GroupNorm + proj_in/out)
# ──────────────────────────────────────────────────────────────────────────────

class Transformer2D(nn.Module):
    def __init__(self, channels: int, heads: int, cross_dim: int, layer_names: List[str],
                 norm_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = [
            Basic2p5DTransformerBlock(channels, heads, cross_dim, layer_name=name)
            for name in layer_names
        ]
        self.proj_out = nn.Linear(channels, channels)

    def __call__(self, x, encoder_hs, num_in_batch, condition_embed_dict,
                 dino_hs, position_voxel_indices, ref_scale, mva_scale):
        input_x = x
        B, H, W, C = x.shape
        x = self.norm(x).reshape(B, H * W, C)
        x = self.proj_in(x)

        for block in self.transformer_blocks:
            cond = condition_embed_dict.get(block.layer_name)
            x = block(x, encoder_hs, num_in_batch, cond, dino_hs,
                      position_voxel_indices, ref_scale, mva_scale)

        x = self.proj_out(x).reshape(B, H, W, C)
        return x + input_x


# ──────────────────────────────────────────────────────────────────────────────
# Down/Up Blocks
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, heads: int,
                 cross_dim: int, num_layers: int, layer_names: List[List[str]],
                 add_downsample: bool = True):
        super().__init__()
        channels = [in_ch] + [out_ch] * (num_layers - 1)
        self.resnets = [ResnetBlock2D(c, out_ch, temb_ch) for c in channels]
        self.attentions = [
            Transformer2D(out_ch, heads, cross_dim, names)
            for names in layer_names
        ]
        if add_downsample:
            self.downsample = Downsample2D(out_ch)
        self.has_downsample = add_downsample

    def __call__(self, x, temb, encoder_hs, num_in_batch, condition_embed_dict,
                 dino_hs, position_voxel_indices, ref_scale, mva_scale):
        states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, temb)
            x = attn(x, encoder_hs, num_in_batch, condition_embed_dict,
                     dino_hs, position_voxel_indices, ref_scale, mva_scale)
            states.append(x)
        if self.has_downsample:
            x = self.downsample(x)
            states.append(x)
        return x, states


class DownBlock2D(nn.Module):
    """Down block without cross-attention."""
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, num_layers: int):
        super().__init__()
        channels = [in_ch] + [out_ch] * (num_layers - 1)
        self.resnets = [ResnetBlock2D(c, out_ch, temb_ch) for c in channels]

    def __call__(self, x, temb, **kwargs):
        states = []
        for resnet in self.resnets:
            x = resnet(x, temb)
            states.append(x)
        return x, states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, prev_ch: int, temb_ch: int,
                 heads: int, cross_dim: int, num_layers: int,
                 layer_names: List[List[str]], add_upsample: bool = True):
        super().__init__()
        channels = [prev_ch + out_ch] + [out_ch + out_ch] * (num_layers - 2) + [out_ch + in_ch]
        if num_layers == 1:
            channels = [prev_ch + in_ch]
        self.resnets = [ResnetBlock2D(c, out_ch, temb_ch) for c in channels]
        self.attentions = [
            Transformer2D(out_ch, heads, cross_dim, names)
            for names in layer_names
        ]
        if add_upsample:
            self.upsample = Upsample2D(out_ch)
        self.has_upsample = add_upsample

    def __call__(self, x, temb, encoder_hs, num_in_batch, condition_embed_dict,
                 dino_hs, position_voxel_indices, ref_scale, mva_scale, residuals):
        for resnet, attn in zip(self.resnets, self.attentions):
            x = mx.concatenate([x, residuals.pop()], axis=-1)
            x = resnet(x, temb)
            x = attn(x, encoder_hs, num_in_batch, condition_embed_dict,
                     dino_hs, position_voxel_indices, ref_scale, mva_scale)
        if self.has_upsample:
            x = self.upsample(x)
        return x


class UpBlock2D(nn.Module):
    """Up block without cross-attention."""
    def __init__(self, in_ch: int, out_ch: int, prev_ch: int, temb_ch: int,
                 num_layers: int, add_upsample: bool = True):
        super().__init__()
        channels = [prev_ch + out_ch] + [out_ch + out_ch] * (num_layers - 2) + [out_ch + in_ch]
        if num_layers == 1:
            channels = [prev_ch + in_ch]
        self.resnets = [ResnetBlock2D(c, out_ch, temb_ch) for c in channels]
        if add_upsample:
            self.upsample = Upsample2D(out_ch)
        self.has_upsample = add_upsample

    def __call__(self, x, temb, residuals, **kwargs):
        for resnet in self.resnets:
            x = mx.concatenate([x, residuals.pop()], axis=-1)
            x = resnet(x, temb)
        if self.has_upsample:
            x = self.upsample(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full Inner UNet
# ──────────────────────────────────────────────────────────────────────────────

class HunyuanInnerUNet(nn.Module):
    """MLX implementation of the Hunyuan3D inner UNet (UNet2DConditionModel).

    Architecture from config.json:
    - in_channels: 12 (latent + normal + position)
    - block_out_channels: [320, 640, 1280, 1280]
    - attention_head_dim: [5, 10, 20, 20] → heads = channels / head_dim
    - cross_attention_dim: 1024
    - layers_per_block: 2
    - down_block_types: CrossAttn, CrossAttn, CrossAttn, DownBlock2D
    - up_block_types: UpBlock2D, CrossAttn, CrossAttn, CrossAttn

    16 transformer blocks total with Basic2p5DTransformerBlock.
    """

    def __init__(self):
        super().__init__()

        channels = [320, 640, 1280, 1280]
        # attention_head_dim in config = number of heads (confusing diffusers naming)
        heads = [5, 10, 20, 20]
        # actual head_dim = channels / heads = 64 for all blocks
        cross_dim = 1024
        temb_ch = channels[0] * 4  # 1280

        # Time embedding
        self.time_embedding = TimestepEmbedding(channels[0], temb_ch)

        # Conv in (12 → 320)
        self.conv_in = nn.Conv2d(12, channels[0], 3, padding=1)

        # Down blocks
        # Layer names match PyTorch: down_{block_i}_{attn_i}_{transformer_i}
        self.down_block_0 = CrossAttnDownBlock2D(
            channels[0], channels[0], temb_ch, heads[0], cross_dim, 2,
            layer_names=[["down_0_0_0"], ["down_0_1_0"]],
            add_downsample=True,
        )
        self.down_block_1 = CrossAttnDownBlock2D(
            channels[0], channels[1], temb_ch, heads[1], cross_dim, 2,
            layer_names=[["down_1_0_0"], ["down_1_1_0"]],
            add_downsample=True,
        )
        self.down_block_2 = CrossAttnDownBlock2D(
            channels[1], channels[2], temb_ch, heads[2], cross_dim, 2,
            layer_names=[["down_2_0_0"], ["down_2_1_0"]],
            add_downsample=True,
        )
        self.down_block_3 = DownBlock2D(channels[2], channels[3], temb_ch, 2)

        # Mid block
        self.mid_resnet_0 = ResnetBlock2D(channels[3], channels[3], temb_ch)
        self.mid_attn = Transformer2D(channels[3], heads[3], cross_dim, ["mid_0_0"])
        self.mid_resnet_1 = ResnetBlock2D(channels[3], channels[3], temb_ch)

        # Up blocks
        # up_block_types: UpBlock2D, CrossAttn, CrossAttn, CrossAttn
        self.up_block_0 = UpBlock2D(
            channels[2], channels[3], channels[3], temb_ch, 3, add_upsample=True,
        )
        self.up_block_1 = CrossAttnUpBlock2D(
            channels[1], channels[2], channels[3], temb_ch, heads[2], cross_dim, 3,
            layer_names=[["up_1_0_0"], ["up_1_1_0"], ["up_1_2_0"]],
            add_upsample=True,
        )
        self.up_block_2 = CrossAttnUpBlock2D(
            channels[0], channels[1], channels[2], temb_ch, heads[1], cross_dim, 3,
            layer_names=[["up_2_0_0"], ["up_2_1_0"], ["up_2_2_0"]],
            add_upsample=True,
        )
        self.up_block_3 = CrossAttnUpBlock2D(
            channels[0], channels[0], channels[1], temb_ch, heads[0], cross_dim, 3,
            layer_names=[["up_3_0_0"], ["up_3_1_0"], ["up_3_2_0"]],
            add_upsample=False,
        )

        # Output
        self.conv_norm_out = nn.GroupNorm(32, channels[0], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(channels[0], 4, 3, padding=1)

    def __call__(
        self,
        sample: mx.array,              # (B*N_pbr*N_gen, C, H, W) NCHW from PyTorch
        timestep: mx.array,            # scalar or (B,)
        encoder_hidden_states: mx.array,  # (B*N_pbr*N_gen, 77, 1024)
        num_in_batch: int = 6,
        condition_embed_dict: Optional[Dict[str, mx.array]] = None,
        dino_hidden_states: Optional[mx.array] = None,
        position_voxel_indices: Optional[dict] = None,
        ref_scale: float = 1.0,
        mva_scale: float = 1.0,
    ) -> mx.array:
        # Convert NCHW → NHWC
        x = sample.transpose(0, 2, 3, 1)

        # Time embedding
        temb = timestep_embedding(
            mx.broadcast_to(timestep.reshape(-1), (x.shape[0],)),
            320,
        ).astype(x.dtype)
        temb = self.time_embedding(temb)

        # Conv in
        x = self.conv_in(x)

        # Common kwargs for attention blocks
        attn_kwargs = dict(
            encoder_hs=encoder_hidden_states,
            num_in_batch=num_in_batch,
            condition_embed_dict=condition_embed_dict or {},
            dino_hs=dino_hidden_states,
            position_voxel_indices=position_voxel_indices,
            ref_scale=ref_scale,
            mva_scale=mva_scale,
        )

        # Down
        residuals = [x]
        x, states = self.down_block_0(x, temb, **attn_kwargs)
        residuals.extend(states)
        x, states = self.down_block_1(x, temb, **attn_kwargs)
        residuals.extend(states)
        x, states = self.down_block_2(x, temb, **attn_kwargs)
        residuals.extend(states)
        x, states = self.down_block_3(x, temb)
        residuals.extend(states)

        # Mid
        x = self.mid_resnet_0(x, temb)
        x = self.mid_attn(x, **attn_kwargs)
        x = self.mid_resnet_1(x, temb)

        # Up
        x = self.up_block_0(x, temb, residuals=residuals)
        x = self.up_block_1(x, temb, **attn_kwargs, residuals=residuals)
        x = self.up_block_2(x, temb, **attn_kwargs, residuals=residuals)
        x = self.up_block_3(x, temb, **attn_kwargs, residuals=residuals)

        # Output
        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        # Convert back NHWC → NCHW
        return x.transpose(0, 3, 1, 2)
