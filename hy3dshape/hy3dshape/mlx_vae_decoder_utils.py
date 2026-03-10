"""MLX dispatch wrapper and weight loading for CrossAttentionDecoder.

Intercepts volume decoding calls to run cross-attention on MLX Metal
instead of PyTorch MPS. Caches KV projection (constant across all
query chunks) for additional speedup.

Usage: called by mesh_gen_native.py via setup_mlx_dispatch_decoder(pipeline).
"""

import time
from typing import Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .mlx_vae_decoder import MLXCrossAttentionDecoder


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pt_to_mx(t) -> mx.array:
    """Convert PyTorch tensor to MLX array (zero-copy via numpy)."""
    import torch
    if isinstance(t, torch.Tensor):
        return mx.array(t.detach().cpu().float().numpy())
    return mx.array(t)


def _tree_map_to_f16(params):
    """Cast all leaf arrays in a nested dict/list to float16."""
    if isinstance(params, mx.array):
        return params.astype(mx.float16) if params.dtype == mx.float32 else params
    elif isinstance(params, dict):
        return {k: _tree_map_to_f16(v) for k, v in params.items()}
    elif isinstance(params, list):
        return [_tree_map_to_f16(v) for v in params]
    return params


def _flatten_dict(d, prefix=""):
    """Flatten nested dict to list of (dotted_key, value) pairs."""
    items = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(_flatten_dict(item, f"{key}.{i}"))
                else:
                    items.append((f"{key}.{i}", item))
        else:
            items.append((key, v))
    return items


# ──────────────────────────────────────────────────────────────────────────────
# Weight Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_decoder_weights_from_pytorch(
    mlx_model: MLXCrossAttentionDecoder,
    pt_decoder,
) -> None:
    """Load weights from PyTorch CrossAttentionDecoder into MLX model.

    The PT model has this structure:
        query_proj.{weight,bias}
        cross_attn_decoder.ln_1.{weight,bias}
        cross_attn_decoder.ln_2.{weight,bias}
        cross_attn_decoder.ln_3.{weight,bias}
        cross_attn_decoder.attn.c_q.{weight}  (no bias, qkv_bias=False)
        cross_attn_decoder.attn.c_kv.{weight}
        cross_attn_decoder.attn.c_proj.{weight,bias}
        cross_attn_decoder.attn.attention.q_norm.{weight,bias}
        cross_attn_decoder.attn.attention.k_norm.{weight,bias}
        cross_attn_decoder.mlp.c_fc.{weight,bias}
        cross_attn_decoder.mlp.c_proj.{weight,bias}
        ln_post.{weight,bias}
        output_proj.{weight,bias}
    """
    t0 = time.time()
    sd = pt_decoder.state_dict()

    weights = {}

    def _copy(src_key, dst_key):
        if src_key in sd:
            weights[dst_key] = _pt_to_mx(sd[src_key])

    # Query projection
    _copy("query_proj.weight", "query_proj.weight")
    _copy("query_proj.bias", "query_proj.bias")

    # Cross-attention block layer norms
    for ln in ["ln_1", "ln_2", "ln_3"]:
        _copy(f"cross_attn_decoder.{ln}.weight", f"{ln}.weight")
        _copy(f"cross_attn_decoder.{ln}.bias", f"{ln}.bias")

    # Attention projections
    for proj in ["c_q", "c_kv", "c_proj"]:
        _copy(f"cross_attn_decoder.attn.{proj}.weight", f"{proj}.weight")
        _copy(f"cross_attn_decoder.attn.{proj}.bias", f"{proj}.bias")

    # QK norms
    _copy("cross_attn_decoder.attn.attention.q_norm.weight", "q_norm.weight")
    _copy("cross_attn_decoder.attn.attention.q_norm.bias", "q_norm.bias")
    _copy("cross_attn_decoder.attn.attention.k_norm.weight", "k_norm.weight")
    _copy("cross_attn_decoder.attn.attention.k_norm.bias", "k_norm.bias")

    # MLP
    _copy("cross_attn_decoder.mlp.c_fc.weight", "mlp_fc.weight")
    _copy("cross_attn_decoder.mlp.c_fc.bias", "mlp_fc.bias")
    _copy("cross_attn_decoder.mlp.c_proj.weight", "mlp_proj.weight")
    _copy("cross_attn_decoder.mlp.c_proj.bias", "mlp_proj.bias")

    # Post-norm + output
    _copy("ln_post.weight", "ln_post.weight")
    _copy("ln_post.bias", "ln_post.bias")
    _copy("output_proj.weight", "output_proj.weight")
    _copy("output_proj.bias", "output_proj.bias")

    # Load into MLX model (strict=False: fourier_embedder.frequencies
    # is computed from config, not loaded from PT weights)
    weight_items = list(weights.items())
    mlx_model.load_weights(weight_items, strict=False)
    mx.eval(mlx_model.parameters())

    elapsed = time.time() - t0
    print(f"  [MLX Decoder] Loaded {len(weight_items)} weight tensors in {elapsed:.1f}s")

    # Verify no missing weights
    model_params = set()
    for k, _ in _flatten_dict(mlx_model.parameters()):
        model_params.add(k)
    loaded = set(k for k, _ in weight_items)
    missing = model_params - loaded
    if missing:
        print(f"  [MLX Decoder] WARNING: {len(missing)} unloaded params: {list(missing)[:5]}...")


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MLXDispatchDecoder:
    """Wraps PyTorch CrossAttentionDecoder, dispatching forward() to MLX.

    Caches KV projection on first call (latents are constant across all
    query chunks in the volume decoding loop).
    """

    def __init__(self, pt_decoder, mlx_decoder: MLXCrossAttentionDecoder,
                 use_fp32: bool = False):
        self._pt = pt_decoder
        self._mlx = mlx_decoder
        self._mlx_dtype = mx.float32 if use_fp32 else mx.float16
        self._kv_cached = False
        self.count = 0  # Match PT decoder's count attribute

    def reset(self):
        """Reset cached state before a new inference run."""
        self._kv_cached = False
        self._mlx._cached_kv = None
        self.count = 0

    def __call__(self, queries=None, query_embeddings=None, latents=None, **kwargs):
        return self.forward(queries, query_embeddings, latents, **kwargs)

    def forward(self, queries=None, query_embeddings=None, latents=None, **kwargs):
        """Intercept forward() and run on MLX Metal."""
        import torch

        # Cache KV on first call
        if not self._kv_cached:
            latents_mx = _pt_to_mx(latents).astype(self._mlx_dtype)
            self._mlx.cache_kv(latents_mx)
            self._kv_cached = True

        # Convert queries to MLX
        if queries is not None:
            queries_mx = _pt_to_mx(queries).astype(self._mlx_dtype)
            self.count += queries_mx.shape[1]
        else:
            # query_embeddings path (not used in VanillaVolumeDecoder)
            raise NotImplementedError("MLX dispatch only supports queries path")

        # Run MLX forward
        output = self._mlx(queries_mx)
        mx.eval(output)

        # Convert back to PyTorch
        output_np = np.array(output)
        output_pt = torch.from_numpy(output_np).to(
            dtype=latents.dtype, device=latents.device
        )

        return output_pt

    def __getattr__(self, name):
        """Fall through to PT decoder for any other attributes."""
        return getattr(self._pt, name)


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_mlx_dispatch_decoder(pipeline) -> Optional[MLXDispatchDecoder]:
    """Set up MLX dispatch on the VAE's geo_decoder.

    Builds MLX model, loads weights from PyTorch, and replaces
    pipeline.vae.geo_decoder with the dispatch wrapper.

    Returns the dispatch wrapper, or None if setup fails.
    """
    import os

    pt_decoder = pipeline.vae.geo_decoder
    use_fp32 = os.environ.get("HUNYUAN_MLX_FP32", "0") == "1"
    dtype_label = "fp32" if use_fp32 else "fp16"

    print(f"  [MLX Decoder] Setting up MLX dispatch ({dtype_label})...")

    # Read config from PT model
    width = pt_decoder.cross_attn_decoder.attn.width
    heads = pt_decoder.cross_attn_decoder.attn.heads
    # Check qkv_bias from c_q
    qkv_bias = pt_decoder.cross_attn_decoder.attn.c_q.bias is not None
    # Check qk_norm
    qk_norm = not isinstance(
        pt_decoder.cross_attn_decoder.attn.attention.q_norm,
        type(pt_decoder)  # nn.Identity check
    )
    # Actually check more robustly
    import torch.nn as torch_nn
    qk_norm = hasattr(pt_decoder.cross_attn_decoder.attn.attention.q_norm, 'weight')

    # Get MLP expand ratio
    mlp_fc = pt_decoder.cross_attn_decoder.mlp.c_fc
    mlp_expand_ratio = mlp_fc.out_features // mlp_fc.in_features

    fourier_out_dim = pt_decoder.fourier_embedder.out_dim

    print(f"  [MLX Decoder] width={width}, heads={heads}, "
          f"mlp_expand={mlp_expand_ratio}, qkv_bias={qkv_bias}, "
          f"qk_norm={qk_norm}, fourier_dim={fourier_out_dim}")

    # Build MLX model
    mlx_model = MLXCrossAttentionDecoder(
        width=width,
        heads=heads,
        mlp_expand_ratio=mlp_expand_ratio,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        fourier_out_dim=fourier_out_dim,
    )

    # Load weights from PyTorch
    load_decoder_weights_from_pytorch(mlx_model, pt_decoder)

    # Cast to target precision
    if not use_fp32:
        params = _tree_map_to_f16(mlx_model.parameters())
        mlx_model.load_weights(list(_flatten_dict(params)), strict=False)
        mx.eval(mlx_model.parameters())

    # Create dispatch wrapper
    dispatch = MLXDispatchDecoder(pt_decoder, mlx_model, use_fp32=use_fp32)

    # Replace pipeline's geo_decoder (bypass nn.Module type check)
    object.__setattr__(pipeline.vae, 'geo_decoder', dispatch)

    print(f"  [MLX Decoder] Ready — volume decoding will run on MLX Metal")
    return dispatch
