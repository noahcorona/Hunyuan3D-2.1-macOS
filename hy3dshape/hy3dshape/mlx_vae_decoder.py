"""MLX implementation of CrossAttentionDecoder for volume decoding.

Port of attention_blocks.py CrossAttentionDecoder to MLX for ~2-3x speedup
on Apple Silicon. The volume decoder is SDPA-dominated (Q: 10K queries x
K: 4096 latents), making it ideal for MLX's 5.1x faster SDPA.

Architecture: FourierEmbedder -> query_proj -> ResidualCrossAttentionBlock
             (cross-attn + MLP) -> ln_post -> output_proj -> (B, N, 1)

Usage: imported by mlx_vae_decoder_utils.py, not used directly.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class MLXFourierEmbedder(nn.Module):
    """Sin/cos positional embedding for 3D query points."""

    def __init__(self, num_freqs: int = 8, input_dim: int = 3,
                 include_input: bool = True, include_pi: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        # Compute frequencies: 2^[0, 1, ..., num_freqs-1]
        freqs = mx.power(2.0, mx.arange(num_freqs, dtype=mx.float32))
        if include_pi:
            freqs = freqs * math.pi
        self.frequencies = freqs  # (num_freqs,)

        temp = 1 if include_input or num_freqs == 0 else 0
        self.out_dim = input_dim * (num_freqs * 2 + temp)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode 3D positions to Fourier features.

        Args:
            x: (..., 3) position coordinates
        Returns:
            (..., out_dim) Fourier embeddings
        """
        if self.num_freqs > 0:
            # (..., 3, 1) * (num_freqs,) -> (..., 3, num_freqs)
            embed = x[..., None] * self.frequencies
            embed = embed.reshape(*x.shape[:-1], -1)  # (..., 3*num_freqs)
            if self.include_input:
                return mx.concatenate([x, mx.sin(embed), mx.cos(embed)], axis=-1)
            else:
                return mx.concatenate([mx.sin(embed), mx.cos(embed)], axis=-1)
        else:
            return x


class MLXCrossAttentionDecoder(nn.Module):
    """MLX CrossAttentionDecoder for volume decoding.

    Single ResidualCrossAttentionBlock with KV caching:
    - KV projection computed once, reused across all query chunks
    - Standard head layout (no interleaving, unlike DiT)
    - QK norm via LayerNorm

    Config (from Hunyuan3D-2.1 VAE):
      width=1024, heads=16, head_dim=64
      mlp_expand_ratio=4, qkv_bias=False, qk_norm=True
      fourier: num_freqs=8, include_pi=False -> out_dim=51
    """

    def __init__(
        self,
        width: int = 1024,
        heads: int = 16,
        mlp_expand_ratio: int = 4,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        fourier_out_dim: int = 51,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fourier embedder (weights computed, not loaded)
        self.fourier_embedder = MLXFourierEmbedder(
            num_freqs=8, input_dim=3, include_input=True, include_pi=False
        )

        # Query projection
        self.query_proj = nn.Linear(fourier_out_dim, width)

        # Cross-attention block
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.ln_2 = nn.LayerNorm(width, eps=1e-6)
        self.ln_3 = nn.LayerNorm(width, eps=1e-6)

        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

        # MLP
        self.mlp_fc = nn.Linear(width, width * mlp_expand_ratio)
        self.mlp_proj = nn.Linear(width * mlp_expand_ratio, width)

        # Post-norm + output
        self.ln_post = nn.LayerNorm(width, eps=1e-6)
        self.output_proj = nn.Linear(width, 1)

        # KV cache (set by dispatch wrapper)
        self._cached_kv = None

    def cache_kv(self, latents: mx.array):
        """Precompute and cache KV projection from latents.

        Called once before the volume decoding loop. The latents are
        constant across all query chunks.
        """
        # Apply ln_2 (data normalization) and KV projection
        normalized = self.ln_2(latents)
        kv = self.c_kv(normalized)  # (B, n_data, 2*width)

        # Split into K, V and reshape for attention
        B, n_data = kv.shape[0], kv.shape[1]
        H, D = self.heads, self.head_dim
        kv = kv.reshape(B, n_data, H, 2 * D)
        k = kv[..., :D]  # (B, n_data, H, D)
        v = kv[..., D:]

        # Apply K norm
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Transpose for SDPA: (B, H, n_data, D)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        self._cached_kv = (k, v)
        mx.eval(k, v)

    def __call__(self, queries: mx.array) -> mx.array:
        """Forward pass for a chunk of query points.

        Args:
            queries: (B, N, 3) 3D coordinates
        Returns:
            (B, N, 1) occupancy logits
        """
        B, N, _ = queries.shape
        H, D = self.heads, self.head_dim

        # Fourier embed + project queries
        query_emb = self.fourier_embedder(queries)
        x = self.query_proj(query_emb)

        # --- Cross-attention ---
        q = self.c_q(self.ln_1(x))  # (B, N, width)
        q = q.reshape(B, N, H, D)

        if self.q_norm is not None:
            q = self.q_norm(q)

        q = q.transpose(0, 2, 1, 3)  # (B, H, N, D)

        # Use cached K, V
        k_cached, v_cached = self._cached_kv

        attn_out = mx.fast.scaled_dot_product_attention(
            q, k_cached, v_cached, scale=self.scale
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, -1)
        attn_out = self.c_proj(attn_out)

        x = x + attn_out

        # --- MLP ---
        mlp_out = self.mlp_proj(nn.gelu(self.mlp_fc(self.ln_3(x))))
        x = x + mlp_out

        # --- Post-norm + output ---
        x = self.ln_post(x)
        return self.output_proj(x)
