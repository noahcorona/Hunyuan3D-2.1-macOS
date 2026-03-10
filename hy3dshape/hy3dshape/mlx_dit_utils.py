"""MLX dispatch wrapper and weight loading for HunYuanDiTPlain.

Loads PyTorch DiT weights into MLX model, wraps forward() to intercept
denoising loop calls and run them on MLX Metal instead of PyTorch MPS.

Usage: called by mesh_gen_native.py via setup_mlx_dispatch_dit(pipeline).
"""

import os
import time
from typing import Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .mlx_dit import MLXHunYuanDiTPlain


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

def load_dit_weights_from_pytorch(mlx_model: MLXHunYuanDiTPlain, pt_model) -> None:
    """Load weights from PyTorch HunYuanDiTPlain into MLX model.

    All layers are Linear (no Conv2d), so weights copy directly.
    Key mapping handles naming differences between PT and MLX models.
    """
    t0 = time.time()
    sd = pt_model.state_dict()

    weights = {}

    def _linear(src_prefix, dst_prefix):
        w_key = f"{src_prefix}.weight"
        b_key = f"{src_prefix}.bias"
        if w_key in sd:
            weights[f"{dst_prefix}.weight"] = _pt_to_mx(sd[w_key])
        if b_key in sd:
            weights[f"{dst_prefix}.bias"] = _pt_to_mx(sd[b_key])

    def _param(src_key, dst_key):
        if src_key in sd:
            weights[dst_key] = _pt_to_mx(sd[src_key])

    # ── Top-level embedders ──
    _linear("x_embedder", "x_embedder")

    # TimestepEmbedder: PT has mlp as Sequential(Linear, GELU, Linear)
    # MLX has mlp_0 and mlp_2
    _linear("t_embedder.mlp.0", "t_embedder.mlp_0")
    _linear("t_embedder.mlp.2", "t_embedder.mlp_2")
    # Timesteps module has no learnable weights

    # ── Blocks ──
    depth = len(mlx_model.blocks)
    for i in range(depth):
        sp = f"blocks.{i}"
        dp = f"blocks.{i}"

        # LayerNorms
        _param(f"{sp}.norm1.weight", f"{dp}.norm1.weight")
        _param(f"{sp}.norm1.bias", f"{dp}.norm1.bias")
        _param(f"{sp}.norm2.weight", f"{dp}.norm2.weight")
        _param(f"{sp}.norm2.bias", f"{dp}.norm2.bias")
        _param(f"{sp}.norm3.weight", f"{dp}.norm3.weight")
        _param(f"{sp}.norm3.bias", f"{dp}.norm3.bias")

        # Self-attention (attn1)
        _linear(f"{sp}.attn1.to_q", f"{dp}.attn1.to_q")
        _linear(f"{sp}.attn1.to_k", f"{dp}.attn1.to_k")
        _linear(f"{sp}.attn1.to_v", f"{dp}.attn1.to_v")
        _linear(f"{sp}.attn1.out_proj", f"{dp}.attn1.out_proj")
        # QK norm (RMSNorm)
        if f"{sp}.attn1.q_norm.weight" in sd:
            _param(f"{sp}.attn1.q_norm.weight", f"{dp}.attn1.q_norm.weight")
            _param(f"{sp}.attn1.k_norm.weight", f"{dp}.attn1.k_norm.weight")

        # Cross-attention (attn2)
        _linear(f"{sp}.attn2.to_q", f"{dp}.attn2.to_q")
        _linear(f"{sp}.attn2.to_k", f"{dp}.attn2.to_k")
        _linear(f"{sp}.attn2.to_v", f"{dp}.attn2.to_v")
        _linear(f"{sp}.attn2.out_proj", f"{dp}.attn2.out_proj")
        if f"{sp}.attn2.q_norm.weight" in sd:
            _param(f"{sp}.attn2.q_norm.weight", f"{dp}.attn2.q_norm.weight")
            _param(f"{sp}.attn2.k_norm.weight", f"{dp}.attn2.k_norm.weight")

        # Skip connection
        if f"{sp}.skip_linear.weight" in sd:
            _linear(f"{sp}.skip_linear", f"{dp}.skip_linear")
            _param(f"{sp}.skip_norm.weight", f"{dp}.skip_norm.weight")
            _param(f"{sp}.skip_norm.bias", f"{dp}.skip_norm.bias")

        # FFN: MLP or MoE
        if f"{sp}.mlp.fc1.weight" in sd:
            # Regular MLP
            _linear(f"{sp}.mlp.fc1", f"{dp}.mlp.fc1")
            _linear(f"{sp}.mlp.fc2", f"{dp}.mlp.fc2")
        elif f"{sp}.moe.gate.weight" in sd:
            # MoE block
            _param(f"{sp}.moe.gate.weight", f"{dp}.moe.gate.weight")
            # Experts: diffusers FeedForward has net[0]=GELU(proj), net[2]=Linear
            num_experts = len([k for k in sd if k.startswith(f"{sp}.moe.experts.") and k.endswith(".net.0.proj.weight")])
            for e in range(num_experts):
                _linear(f"{sp}.moe.experts.{e}.net.0.proj", f"{dp}.moe.experts.{e}.proj_in")
                _linear(f"{sp}.moe.experts.{e}.net.2", f"{dp}.moe.experts.{e}.proj_out")
            # Shared expert
            _linear(f"{sp}.moe.shared_experts.net.0.proj", f"{dp}.moe.shared_experts.proj_in")
            _linear(f"{sp}.moe.shared_experts.net.2", f"{dp}.moe.shared_experts.proj_out")

    # ── Final layer ──
    _param("final_layer.norm_final.weight", "final_layer.norm_final.weight")
    _param("final_layer.norm_final.bias", "final_layer.norm_final.bias")
    _linear("final_layer.linear", "final_layer.linear")

    # Load into MLX model
    weight_items = list(weights.items())
    mlx_model.load_weights(weight_items)
    mx.eval(mlx_model.parameters())

    elapsed = time.time() - t0
    print(f"  [MLX DiT] Loaded {len(weight_items)} weight tensors in {elapsed:.1f}s")

    # Verify no missing weights
    model_params = set()
    for k, _ in _flatten_dict(mlx_model.parameters()):
        model_params.add(k)
    loaded = set(k for k, _ in weight_items)
    missing = model_params - loaded
    if missing:
        print(f"  [MLX DiT] WARNING: {len(missing)} unloaded params: {list(missing)[:5]}...")


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MLXDispatchDiT:
    """Wraps PyTorch DiT model, dispatching forward() calls to MLX.

    Caches conditioning (constant across all denoising steps) on first call.
    All steps run on MLX — no PyTorch fallback needed.
    """

    def __init__(self, pt_model, mlx_model: MLXHunYuanDiTPlain,
                 use_fp32: bool = False):
        self._pt_model = pt_model
        self._mlx_model = mlx_model
        self._mlx_dtype = mx.float32 if use_fp32 else mx.float16
        self._cached_cond = None
        self._step = 0

        # Copy attributes from PT model that pipeline might access
        for attr in ("device", "dtype", "in_channels", "out_channels",
                     "hidden_size", "depth", "input_size"):
            if hasattr(pt_model, attr):
                setattr(self, attr, getattr(pt_model, attr))

    def reset(self):
        """Reset cached state before a new inference run."""
        self._cached_cond = None
        self._step = 0

    def __call__(self, x, t, contexts, **kwargs):
        return self.forward(x, t, contexts, **kwargs)

    def forward(self, x, t, contexts, **kwargs):
        """Intercept forward() and run on MLX Metal."""
        import torch

        self._step += 1

        # Cache conditioning on first call (constant across steps)
        if self._cached_cond is None:
            cond_pt = contexts["main"]
            self._cached_cond = _pt_to_mx(cond_pt).astype(self._mlx_dtype)
            mx.eval(self._cached_cond)

        # Convert inputs to MLX
        x_mx = _pt_to_mx(x).astype(self._mlx_dtype)
        t_mx = _pt_to_mx(t).astype(self._mlx_dtype)

        # Run MLX forward
        output = self._mlx_model(x_mx, t_mx, self._cached_cond)

        # Force Metal GPU completion before converting back
        mx.eval(output)

        # Convert back to PyTorch
        output_np = np.array(output)
        output_pt = torch.from_numpy(output_np).to(dtype=x.dtype, device=x.device)

        return output_pt

    def __getattr__(self, name):
        """Fall through to PT model for any other attributes."""
        return getattr(self._pt_model, name)


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_mlx_dispatch_dit(pipeline) -> Optional[MLXDispatchDiT]:
    """Set up MLX dispatch on a HunYuanDiTPlain pipeline.

    Builds MLX model matching the PyTorch model's config, loads weights,
    and replaces pipeline.model with the dispatch wrapper.

    Returns the dispatch wrapper, or None if MLX setup fails.
    """
    pt_model = pipeline.model
    use_fp32 = os.environ.get("HUNYUAN_MLX_FP32", "0") == "1"
    dtype_label = "fp32" if use_fp32 else "fp16"

    print(f"  [MLX DiT] Setting up MLX dispatch ({dtype_label})...")

    # Read config from PyTorch model
    depth = pt_model.depth
    config = {
        "input_size": getattr(pt_model, "input_size", 4096),
        "in_channels": getattr(pt_model, "in_channels", 64),
        "hidden_size": getattr(pt_model, "hidden_size", 2048),
        "context_dim": getattr(pt_model, "context_dim", 1024),
        "depth": depth,
        "num_heads": getattr(pt_model, "num_heads", 16),
        "qk_norm": True,  # From config.yaml
        "qkv_bias": False,  # From config.yaml
    }

    # Detect MoE config from the PT model blocks
    num_moe_layers = 0
    num_experts = 8
    moe_top_k = 2
    for block in pt_model.blocks:
        if hasattr(block, 'use_moe') and block.use_moe:
            num_moe_layers += 1
            if hasattr(block, 'moe'):
                num_experts = len(block.moe.experts)
                moe_top_k = block.moe.moe_top_k

    config["num_moe_layers"] = num_moe_layers
    config["num_experts"] = num_experts
    config["moe_top_k"] = moe_top_k

    print(f"  [MLX DiT] Model: {depth} blocks ({num_moe_layers} MoE), "
          f"hidden={config['hidden_size']}, heads={config['num_heads']}, "
          f"experts={num_experts}x{moe_top_k}")

    # Build MLX model
    mlx_model = MLXHunYuanDiTPlain(**config)

    # Load weights from PyTorch
    load_dit_weights_from_pytorch(mlx_model, pt_model)

    # Cast to target precision
    if not use_fp32:
        params = _tree_map_to_f16(mlx_model.parameters())
        mlx_model.load_weights(list(_flatten_dict(params)))
        mx.eval(mlx_model.parameters())

    # Create dispatch wrapper
    dispatch = MLXDispatchDiT(pt_model, mlx_model, use_fp32=use_fp32)

    # Replace pipeline model
    pipeline.model = dispatch

    print(f"  [MLX DiT] Ready — all denoising steps will run on MLX Metal")
    return dispatch
