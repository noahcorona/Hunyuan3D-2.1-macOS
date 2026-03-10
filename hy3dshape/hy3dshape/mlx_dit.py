"""MLX implementation of HunYuanDiTPlain for mesh generation.

Port of hunyuandit.py (PyTorch) to MLX for ~2-3x speedup on Apple Silicon.
Architecture: 21 HunYuanDiTBlocks with U-Net skip connections, MoE in last 6.

Usage: imported by mlx_dit_utils.py, not used directly.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────────

class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings matching PyTorch Timesteps."""
    def __init__(self, num_channels: int, downscale_freq_shift: float = 0.0,
                 scale: int = 1, max_period: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def __call__(self, timesteps: mx.array) -> mx.array:
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * mx.arange(half_dim, dtype=mx.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = mx.exp(exponent)
        emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
        emb = self.scale * emb
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return emb


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256,
                 out_size: Optional[int] = None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp_0 = nn.Linear(hidden_size, frequency_embedding_size)
        self.mlp_2 = nn.Linear(frequency_embedding_size, out_size)
        self.time_embed = Timesteps(hidden_size)

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self.time_embed(t)
        t_emb = self.mlp_2(nn.gelu(self.mlp_0(t_freq)))
        return t_emb[:, None, :]  # (B, 1, D)


class MLP(nn.Module):
    """Simple MLP: Linear -> GELU -> Linear."""
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class FeedForwardGELU(nn.Module):
    """FeedForward with GELU activation (used in MoE experts).

    Matches diffusers FeedForward(activation_fn="gelu"):
      GELU(Linear(dim, inner_dim)) -> Linear(inner_dim, dim)
    """
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.proj_in = nn.Linear(dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj_out(nn.gelu(self.proj_in(x)))


# ──────────────────────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Self-attention with separate Q, K, V projections and optional QK norm."""
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False,
                 qk_norm: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.to_q(x)  # (B, N, C)
        k = self.to_k(x)
        v = self.to_v(x)

        # Match PyTorch's interleaved head arrangement:
        # cat(q,k,v) -> view(1, B*N, H, 3*D) -> split -> reshape(B, N, H, D)
        # This scrambles features across heads — trained weights expect it.
        qkv = mx.concatenate([q, k, v], axis=-1)  # (B, N, 3*C)
        qkv = qkv.reshape(B * N, H, 3 * D)  # (B*N, H, 3*D)
        q = qkv[:, :, :D].reshape(B, N, H, D)
        k = qkv[:, :, D:2*D].reshape(B, N, H, D)
        v = qkv[:, :, 2*D:].reshape(B, N, H, D)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # (B, N, H, D) -> (B, H, N, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: Q from x, K/V from context."""
    def __init__(self, qdim: int, kdim: int, num_heads: int,
                 qkv_bias: bool = False, qk_norm: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = qdim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.out_proj = nn.Linear(qdim, qdim)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        B, S1, _ = x.shape
        S2 = y.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.to_q(x)  # (B, S1, qdim)
        k = self.to_k(y)  # (B, S2, qdim)
        v = self.to_v(y)

        # Q: standard reshape (matches PT which uses view directly)
        q = q.reshape(B, S1, H, D)

        # K/V: interleaved head arrangement (matches PT cat+view+split)
        kv = mx.concatenate([k, v], axis=-1)  # (B, S2, 2*qdim)
        kv = kv.reshape(B * S2, H, 2 * D)
        k = kv[:, :, :D].reshape(B, S2, H, D)
        v = kv[:, :, D:].reshape(B, S2, H, D)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, S1, -1)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# MoE
# ──────────────────────────────────────────────────────────────────────────────

class MoEGate(nn.Module):
    """Top-k expert routing gate."""
    def __init__(self, embed_dim: int, num_experts: int = 8,
                 num_experts_per_tok: int = 2):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.weight = mx.zeros((num_experts, embed_dim))

    def __call__(self, hidden_states: mx.array):
        B, S, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)
        logits = x @ self.weight.T
        scores = mx.softmax(logits, axis=-1)
        # Top-k selection
        topk_idx = mx.argpartition(-scores, kth=self.top_k, axis=-1)[:, :self.top_k]
        # Gather weights for selected experts
        topk_weight = mx.take_along_axis(scores, topk_idx, axis=-1)
        return topk_idx, topk_weight


class MoEBlock(nn.Module):
    """Mixture of Experts block with shared expert and proper token routing.

    Uses sort-based routing: tokens are sorted by expert assignment so each
    expert only processes its routed tokens (top-2 out of 8 = 4x less compute
    than the naive all-experts-on-all-tokens approach).
    """
    def __init__(self, dim: int, num_experts: int = 8, moe_top_k: int = 2,
                 ff_inner_dim: Optional[int] = None):
        super().__init__()
        self.moe_top_k = moe_top_k
        self.num_experts = num_experts
        if ff_inner_dim is None:
            ff_inner_dim = dim * 4

        self.experts = [FeedForwardGELU(dim, ff_inner_dim) for _ in range(num_experts)]
        self.gate = MoEGate(dim, num_experts, moe_top_k)
        self.shared_experts = FeedForwardGELU(dim, ff_inner_dim)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        orig_shape = hidden_states.shape
        B_S = orig_shape[0] * orig_shape[1]
        D = orig_shape[-1]
        x = hidden_states.reshape(B_S, D)

        topk_idx, topk_weight = self.gate(hidden_states)  # (B*S, top_k) each

        # Flatten top-k: each token appears top_k times
        flat_expert_idx = topk_idx.reshape(-1)          # (B*S*top_k,)
        flat_weight = topk_weight.reshape(-1, 1)        # (B*S*top_k, 1)
        token_ids = mx.repeat(mx.arange(B_S), self.moe_top_k)  # (B*S*top_k,)

        # Sort by expert index to group tokens per expert
        sort_perm = mx.argsort(flat_expert_idx)
        sorted_token_ids = token_ids[sort_perm]
        sorted_weight = flat_weight[sort_perm]

        # Count tokens per expert (single sync for all counts)
        counts_list = [mx.sum(flat_expert_idx == i) for i in range(self.num_experts)]
        count_arr = mx.stack(counts_list)
        mx.eval(count_arr)
        counts = [int(c) for c in count_arr.tolist()]

        # Process each expert only on its routed tokens
        chunks = []
        offset = 0
        for i in range(self.num_experts):
            c = counts[i]
            if c > 0:
                expert_input = x[sorted_token_ids[offset:offset + c]]
                expert_result = self.experts[i](expert_input)
                chunks.append(expert_result * sorted_weight[offset:offset + c])
            offset += c

        sorted_output = mx.concatenate(chunks, axis=0)

        # Un-sort back to original token order and sum over top-k slots
        inv_perm = mx.argsort(sort_perm)
        flat_output = sorted_output[inv_perm]
        expert_out = flat_output.reshape(B_S, self.moe_top_k, D).sum(axis=1)

        y = expert_out.reshape(*orig_shape)
        y = y + self.shared_experts(x).reshape(*orig_shape)
        return y


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────────────────────────────────────

class HunYuanDiTBlock(nn.Module):
    """Single block: self-attn -> cross-attn -> MLP/MoE, with optional skip connection."""
    def __init__(self, hidden_size: int, num_heads: int,
                 context_dim: int = 1024, qkv_bias: bool = False,
                 qk_norm: bool = False, skip_connection: bool = False,
                 use_moe: bool = False, num_experts: int = 8,
                 moe_top_k: int = 2):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads, qkv_bias=qkv_bias,
                               qk_norm=qk_norm)

        # Cross-attention
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, context_dim, num_heads,
                                    qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)

        # Skip connection
        self.skip_connection = skip_connection
        if skip_connection:
            self.skip_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)

        # FFN
        self.use_moe = use_moe
        if use_moe:
            self.moe = MoEBlock(hidden_size, num_experts=num_experts,
                                moe_top_k=moe_top_k,
                                ff_inner_dim=int(hidden_size * 4.0))
        else:
            self.mlp = MLP(hidden_size)

    def __call__(self, x: mx.array, cond: mx.array,
                 skip_value: Optional[mx.array] = None) -> mx.array:
        if self.skip_connection and skip_value is not None:
            cat = mx.concatenate([skip_value, x], axis=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # Self-attention
        x = x + self.attn1(self.norm1(x))

        # Cross-attention
        x = x + self.attn2(self.norm2(x), cond)

        # FFN
        if self.use_moe:
            x = x + self.moe(self.norm3(x))
        else:
            x = x + self.mlp(self.norm3(x))

        return x


# ──────────────────────────────────────────────────────────────────────────────
# Final Layer
# ──────────────────────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    """LayerNorm -> strip first token -> Linear."""
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm_final(x)
        x = x[:, 1:]  # Strip prepended conditioning token
        return self.linear(x)


# ──────────────────────────────────────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────────────────────────────────────

class MLXHunYuanDiTPlain(nn.Module):
    """MLX implementation of HunYuanDiTPlain for mesh generation.

    Architecture: 21 blocks with U-Net skip connections.
    First half (0..depth//2) saves skip values.
    Second half (depth//2+1..depth-1) consumes them.
    Last num_moe_layers blocks use Mixture of Experts.
    """

    def __init__(
        self,
        input_size: int = 4096,
        in_channels: int = 64,
        hidden_size: int = 2048,
        context_dim: int = 1024,
        depth: int = 21,
        num_heads: int = 16,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Embedders
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4)

        # Blocks
        self.blocks = [
            HunYuanDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                context_dim=context_dim,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                skip_connection=(layer > depth // 2),
                use_moe=(depth - layer <= num_moe_layers),
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
            for layer in range(depth)
        ]

        self.final_layer = FinalLayer(hidden_size, in_channels)

    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Latent tokens (B, 4096, 64)
            t: Timesteps (B,) — raw integer timesteps
            cond: Conditioning tokens (B, N_tokens, context_dim)
                  Already projected by conditioner, NOT through any model layer.
        """
        c = self.t_embedder(t)  # (B, 1, hidden_size)
        x = self.x_embedder(x)  # (B, 4096, hidden_size)

        # Prepend conditioning token
        x = mx.concatenate([c, x], axis=1)  # (B, 4097, hidden_size)

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        x = self.final_layer(x)  # Strips first token, projects
        return x
