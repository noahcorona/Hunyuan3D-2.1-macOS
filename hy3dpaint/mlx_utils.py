"""MLX dispatch wrapper and weight loader for Hunyuan3D inner UNet.

Provides MLXDispatchUNet: a drop-in replacement for the inner UNet that
runs generation passes (steps 2-15) on MLX for ~3-5x speedup over PyTorch MPS.

Usage in textureGenPipeline.py:
    from mlx_utils import setup_mlx_dispatch
    setup_mlx_dispatch(self.models["multiview_model"].pipeline)

Note: mx.eval() is MLX's graph evaluation function — it forces lazy computation
to complete on the Metal GPU. It is NOT Python's built-in eval().
"""

import os
import time
from typing import Dict, Optional

import numpy as np
import torch

import mlx.core as mx
import mlx.nn as mlx_nn

try:
    from .mlx_unet import HunyuanInnerUNet
except ImportError:
    from mlx_unet import HunyuanInnerUNet


# ──────────────────────────────────────────────────────────────────────────────
# Weight Loading
# ──────────────────────────────────────────────────────────────────────────────

def _pt_to_mx(tensor: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array, preserving dtype."""
    return mx.array(tensor.cpu().numpy())


def _conv_weight_pt_to_mlx(w: torch.Tensor) -> mx.array:
    """Convert Conv2d weight from PyTorch OIHW to MLX OHWI."""
    return mx.array(w.cpu().permute(0, 2, 3, 1).numpy())


def _load_attention_weights(block, prefix: str, state_dict: dict, dim: int):
    """Load weights for a Basic2p5DTransformerBlock from PyTorch state dict."""
    p = prefix

    def get(key):
        return _pt_to_mx(state_dict[key])

    # Self-attention (albedo) — under transformer.attn1.*
    block.attn1_to_q.weight = get(f"{p}.transformer.attn1.to_q.weight")
    block.attn1_to_k.weight = get(f"{p}.transformer.attn1.to_k.weight")
    block.attn1_to_v.weight = get(f"{p}.transformer.attn1.to_v.weight")
    block.attn1_to_out.weight = get(f"{p}.transformer.attn1.to_out.0.weight")
    block.attn1_to_out.bias = get(f"{p}.transformer.attn1.to_out.0.bias")

    # Self-attention (MR) — under transformer.attn1.processor.*
    block.attn1_to_q_mr.weight = get(f"{p}.transformer.attn1.processor.to_q_mr.weight")
    block.attn1_to_k_mr.weight = get(f"{p}.transformer.attn1.processor.to_k_mr.weight")
    block.attn1_to_v_mr.weight = get(f"{p}.transformer.attn1.processor.to_v_mr.weight")
    block.attn1_to_out_mr.weight = get(f"{p}.transformer.attn1.processor.to_out_mr.0.weight")
    block.attn1_to_out_mr.bias = get(f"{p}.transformer.attn1.processor.to_out_mr.0.bias")

    # Reference attention
    block.attn_ref_to_q.weight = get(f"{p}.attn_refview.to_q.weight")
    block.attn_ref_to_k.weight = get(f"{p}.attn_refview.to_k.weight")
    block.attn_ref_to_v.weight = get(f"{p}.attn_refview.to_v.weight")
    block.attn_ref_to_out.weight = get(f"{p}.attn_refview.to_out.0.weight")
    block.attn_ref_to_out.bias = get(f"{p}.attn_refview.to_out.0.bias")
    block.attn_ref_to_v_mr.weight = get(f"{p}.attn_refview.processor.to_v_mr.weight")
    block.attn_ref_to_out_mr.weight = get(f"{p}.attn_refview.processor.to_out_mr.0.weight")
    block.attn_ref_to_out_mr.bias = get(f"{p}.attn_refview.processor.to_out_mr.0.bias")

    # Multiview attention
    block.attn_mv_to_q.weight = get(f"{p}.attn_multiview.to_q.weight")
    block.attn_mv_to_k.weight = get(f"{p}.attn_multiview.to_k.weight")
    block.attn_mv_to_v.weight = get(f"{p}.attn_multiview.to_v.weight")
    block.attn_mv_to_out.weight = get(f"{p}.attn_multiview.to_out.0.weight")
    block.attn_mv_to_out.bias = get(f"{p}.attn_multiview.to_out.0.bias")

    # Text cross-attention
    block.attn2_to_q.weight = get(f"{p}.transformer.attn2.to_q.weight")
    block.attn2_to_k.weight = get(f"{p}.transformer.attn2.to_k.weight")
    block.attn2_to_v.weight = get(f"{p}.transformer.attn2.to_v.weight")
    block.attn2_to_out.weight = get(f"{p}.transformer.attn2.to_out.0.weight")
    block.attn2_to_out.bias = get(f"{p}.transformer.attn2.to_out.0.bias")

    # DINO cross-attention
    block.attn_dino_to_q.weight = get(f"{p}.attn_dino.to_q.weight")
    block.attn_dino_to_k.weight = get(f"{p}.attn_dino.to_k.weight")
    block.attn_dino_to_v.weight = get(f"{p}.attn_dino.to_v.weight")
    block.attn_dino_to_out.weight = get(f"{p}.attn_dino.to_out.0.weight")
    block.attn_dino_to_out.bias = get(f"{p}.attn_dino.to_out.0.bias")

    # Norms
    block.norm1.weight = get(f"{p}.transformer.norm1.weight")
    block.norm1.bias = get(f"{p}.transformer.norm1.bias")
    block.norm2.weight = get(f"{p}.transformer.norm2.weight")
    block.norm2.bias = get(f"{p}.transformer.norm2.bias")
    block.norm3.weight = get(f"{p}.transformer.norm3.weight")
    block.norm3.bias = get(f"{p}.transformer.norm3.bias")

    # FFN (GEGLU)
    block.ff_proj.weight = get(f"{p}.transformer.ff.net.0.proj.weight")
    block.ff_proj.bias = get(f"{p}.transformer.ff.net.0.proj.bias")
    block.ff_out.weight = get(f"{p}.transformer.ff.net.2.weight")
    block.ff_out.bias = get(f"{p}.transformer.ff.net.2.bias")


def _load_resnet_weights(resnet, prefix: str, state_dict: dict):
    """Load ResnetBlock2D weights."""
    def get(key):
        return _pt_to_mx(state_dict[key])
    def get_conv(key):
        return _conv_weight_pt_to_mlx(state_dict[key])

    resnet.norm1.weight = get(f"{prefix}.norm1.weight")
    resnet.norm1.bias = get(f"{prefix}.norm1.bias")
    resnet.conv1.weight = get_conv(f"{prefix}.conv1.weight")
    resnet.conv1.bias = get(f"{prefix}.conv1.bias")
    resnet.time_emb_proj.weight = get(f"{prefix}.time_emb_proj.weight")
    resnet.time_emb_proj.bias = get(f"{prefix}.time_emb_proj.bias")
    resnet.norm2.weight = get(f"{prefix}.norm2.weight")
    resnet.norm2.bias = get(f"{prefix}.norm2.bias")
    resnet.conv2.weight = get_conv(f"{prefix}.conv2.weight")
    resnet.conv2.bias = get(f"{prefix}.conv2.bias")

    shortcut_key = f"{prefix}.conv_shortcut.weight"
    if shortcut_key in state_dict:
        resnet.conv_shortcut.weight = get_conv(shortcut_key)
        resnet.conv_shortcut.bias = get(f"{prefix}.conv_shortcut.bias")


def _load_transformer2d_weights(transformer2d, prefix: str, state_dict: dict):
    """Load Transformer2D (norm + proj_in + blocks + proj_out)."""
    def get(key):
        return _pt_to_mx(state_dict[key])

    transformer2d.norm.weight = get(f"{prefix}.norm.weight")
    transformer2d.norm.bias = get(f"{prefix}.norm.bias")
    transformer2d.proj_in.weight = get(f"{prefix}.proj_in.weight")
    transformer2d.proj_in.bias = get(f"{prefix}.proj_in.bias")
    transformer2d.proj_out.weight = get(f"{prefix}.proj_out.weight")
    transformer2d.proj_out.bias = get(f"{prefix}.proj_out.bias")

    for i, block in enumerate(transformer2d.transformer_blocks):
        _load_attention_weights(
            block, f"{prefix}.transformer_blocks.{i}", state_dict, block.dim
        )


def load_weights_from_pytorch(model: HunyuanInnerUNet, ckpt_path: str):
    """Load all weights from PyTorch checkpoint into MLX model."""
    print(f"  Loading PyTorch checkpoint: {ckpt_path}")
    t0 = time.time()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"  Checkpoint loaded in {time.time() - t0:.1f}s ({len(state_dict)} keys)")

    def get(key):
        return _pt_to_mx(state_dict[key])
    def get_conv(key):
        return _conv_weight_pt_to_mlx(state_dict[key])

    # Strip "unet." prefix if present
    if any(k.startswith("unet.") for k in state_dict):
        state_dict = {k.replace("unet.", "", 1): v for k, v in state_dict.items()}

    t0 = time.time()

    # Time embedding
    model.time_embedding.linear_1.weight = get("time_embedding.linear_1.weight")
    model.time_embedding.linear_1.bias = get("time_embedding.linear_1.bias")
    model.time_embedding.linear_2.weight = get("time_embedding.linear_2.weight")
    model.time_embedding.linear_2.bias = get("time_embedding.linear_2.bias")

    # Conv in/out
    model.conv_in.weight = get_conv("conv_in.weight")
    model.conv_in.bias = get("conv_in.bias")
    model.conv_norm_out.weight = get("conv_norm_out.weight")
    model.conv_norm_out.bias = get("conv_norm_out.bias")
    model.conv_out.weight = get_conv("conv_out.weight")
    model.conv_out.bias = get("conv_out.bias")

    # Down blocks
    for block_idx, block in enumerate([model.down_block_0, model.down_block_1,
                                        model.down_block_2]):
        prefix = f"down_blocks.{block_idx}"
        for i, resnet in enumerate(block.resnets):
            _load_resnet_weights(resnet, f"{prefix}.resnets.{i}", state_dict)
        for i, attn in enumerate(block.attentions):
            _load_transformer2d_weights(attn, f"{prefix}.attentions.{i}", state_dict)
        if block.has_downsample:
            ds_prefix = f"{prefix}.downsamplers.0.conv"
            block.downsample.conv.weight = get_conv(f"{ds_prefix}.weight")
            block.downsample.conv.bias = get(f"{ds_prefix}.bias")

    # Down block 3 (no attention)
    for i, resnet in enumerate(model.down_block_3.resnets):
        _load_resnet_weights(resnet, f"down_blocks.3.resnets.{i}", state_dict)

    # Mid block
    _load_resnet_weights(model.mid_resnet_0, "mid_block.resnets.0", state_dict)
    _load_resnet_weights(model.mid_resnet_1, "mid_block.resnets.1", state_dict)
    _load_transformer2d_weights(model.mid_attn, "mid_block.attentions.0", state_dict)

    # Up blocks
    # up_block_0: UpBlock2D (no attention)
    for i, resnet in enumerate(model.up_block_0.resnets):
        _load_resnet_weights(resnet, f"up_blocks.0.resnets.{i}", state_dict)

    # up_blocks 1-3: CrossAttnUpBlock2D
    for block_idx, block in enumerate([model.up_block_1, model.up_block_2,
                                        model.up_block_3], start=1):
        prefix = f"up_blocks.{block_idx}"
        for i, resnet in enumerate(block.resnets):
            _load_resnet_weights(resnet, f"{prefix}.resnets.{i}", state_dict)
        for i, attn in enumerate(block.attentions):
            _load_transformer2d_weights(attn, f"{prefix}.attentions.{i}", state_dict)
        if block.has_upsample:
            us_prefix = f"{prefix}.upsamplers.0.conv"
            block.upsample.conv.weight = get_conv(f"{us_prefix}.weight")
            block.upsample.conv.bias = get(f"{us_prefix}.bias")

    print(f"  Weights mapped in {time.time() - t0:.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MLXDispatchUNet(torch.nn.Module):
    """Drop-in replacement for inner UNet that dispatches gen passes to MLX.

    Step 1: PyTorch (builds caches in outer UNet).
    Steps 2-15: MLX (5x faster SDPA on Metal).
    """

    def __init__(self, original_unet, mlx_model: HunyuanInnerUNet, use_fp32: bool = False):
        super().__init__()
        self._original = original_unet
        self._mlx_model = mlx_model
        self._use_fp32 = use_fp32
        self._mlx_dtype = mx.float32 if use_fp32 else mx.float16
        self._gen_call_count = 0
        self._cached_mlx = None  # Pre-converted MLX arrays

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._original, name)

    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs", {}) or {}
        mode = cross_attention_kwargs.get("mode", "")

        # Reference write pass — always PyTorch
        if "w" in mode:
            return self._original(sample, timestep, encoder_hidden_states, *args, **kwargs)

        self._gen_call_count += 1

        # First gen pass — PyTorch (populates caches in outer UNet)
        if self._gen_call_count == 1:
            self._cache_constant_inputs(encoder_hidden_states, kwargs)
            return self._original(sample, timestep, encoder_hidden_states, *args, **kwargs)

        # Steps 2-15 — MLX
        if os.environ.get("HUNYUAN_MLX_COMPARE"):
            return self._compare_forward(sample, timestep, encoder_hidden_states, args, kwargs)
        return self._mlx_forward(sample, timestep)

    def _compare_forward(self, sample, timestep, encoder_hidden_states, args, kwargs):
        """Run both PyTorch and MLX, compare outputs numerically."""
        # Print input stats
        s = sample.detach().cpu().float().numpy()
        print(f"\n  [MLX-COMPARE] Input sample: range=[{s.min():.4f}, {s.max():.4f}], "
              f"mean={s.mean():.4f}, dtype={sample.dtype}, shape={list(sample.shape)}")
        print(f"  [MLX-COMPARE] Timestep: {timestep.item():.1f}")

        # PyTorch MPS
        pt_out = self._original(sample, timestep, encoder_hidden_states, *args, **kwargs)
        pt_arr = pt_out[0].detach().cpu().float().numpy()

        # MLX
        mlx_out = self._mlx_forward(sample, timestep)
        mlx_arr = mlx_out[0].detach().cpu().float().numpy()

        # Compare
        abs_diff = np.abs(pt_arr - mlx_arr)
        rel_diff = abs_diff / (np.abs(pt_arr) + 1e-8)
        print(f"\n  [MLX-COMPARE] Step {self._gen_call_count}:")
        print(f"    PT  range: [{pt_arr.min():.4f}, {pt_arr.max():.4f}], mean={pt_arr.mean():.4f}, std={pt_arr.std():.4f}")
        print(f"    MLX range: [{mlx_arr.min():.4f}, {mlx_arr.max():.4f}], mean={mlx_arr.mean():.4f}, std={mlx_arr.std():.4f}")
        print(f"    Abs diff:  mean={abs_diff.mean():.6f}, max={abs_diff.max():.6f}, p99={np.percentile(abs_diff, 99):.6f}")
        print(f"    Rel diff:  mean={rel_diff.mean():.6f}, max={rel_diff.max():.6f}, p99={np.percentile(rel_diff, 99):.6f}")

        # Use PyTorch result (ground truth)
        return pt_out

    def _cache_constant_inputs(self, encoder_hidden_states, kwargs):
        """Convert constant tensors to MLX arrays (cached for steps 2-15)."""
        cross_attn = kwargs.get("cross_attention_kwargs") or {}
        dtype = self._mlx_dtype

        def to_mlx(tensor):
            return _pt_to_mx(tensor).astype(dtype)

        cached = {}
        cached["encoder_hs"] = to_mlx(encoder_hidden_states)
        cached["num_in_batch"] = cross_attn.get("num_in_batch", 6)
        # ref_scale/mva_scale may be float, scalar tensor, or multi-element tensor
        ref_scale = cross_attn.get("ref_scale", 1.0)
        mva_scale = cross_attn.get("mva_scale", 1.0)
        if isinstance(ref_scale, torch.Tensor):
            cached["ref_scale"] = to_mlx(ref_scale) if ref_scale.numel() > 1 else ref_scale.item()
        else:
            cached["ref_scale"] = float(ref_scale)
        if isinstance(mva_scale, torch.Tensor):
            cached["mva_scale"] = to_mlx(mva_scale) if mva_scale.numel() > 1 else mva_scale.item()
        else:
            cached["mva_scale"] = float(mva_scale)

        # DINO hidden states
        dino = cross_attn.get("dino_hidden_states")
        if dino is not None:
            cached["dino_hs"] = to_mlx(dino)

        # Condition embed dict
        cond_dict = cross_attn.get("condition_embed_dict", {})
        cached["condition_embed_dict"] = {
            name: to_mlx(tensor) for name, tensor in cond_dict.items()
        }

        # Position voxel indices (int32, no upcast needed)
        pvox = cross_attn.get("position_voxel_indices", {})
        cached["position_voxel_indices"] = {}
        for key, entry in pvox.items():
            cached["position_voxel_indices"][key] = {
                "voxel_indices": _pt_to_mx(entry["voxel_indices"]),
                "voxel_resolution": entry["voxel_resolution"],
            }

        self._cached_mlx = cached
        precision = "fp32" if self._use_fp32 else "fp16"
        print(f"  [MLX] Cached {len(cached)} inputs for steps 2-15 ({precision})")

    def _mlx_forward(self, sample, timestep):
        """Run MLX model and convert output back to PyTorch."""
        c = self._cached_mlx
        dtype = self._mlx_dtype

        sample_mlx = _pt_to_mx(sample).astype(dtype)
        timestep_mlx = mx.array([timestep.item()]).astype(dtype)

        # Run MLX UNet
        output = self._mlx_model(
            sample_mlx,
            timestep_mlx,
            c["encoder_hs"],
            num_in_batch=c["num_in_batch"],
            condition_embed_dict=c["condition_embed_dict"],
            dino_hidden_states=c.get("dino_hs"),
            position_voxel_indices=c["position_voxel_indices"],
            ref_scale=c["ref_scale"],
            mva_scale=c["mva_scale"],
        )

        # mx.eval forces lazy computation to complete on Metal GPU
        mx.eval(output)  # noqa: S307 — this is MLX's graph eval, not Python eval

        # Convert back to PyTorch on MPS
        output_pt = torch.from_numpy(np.array(output)).to(
            dtype=sample.dtype, device=sample.device
        )

        # Return as tuple (matching inner UNet return format)
        return (output_pt,)

    def reset(self):
        """Reset for new denoising run."""
        self._gen_call_count = 0
        self._cached_mlx = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tree_map_to_f16(tree):
    """Recursively convert all mx.arrays in a nested dict/list to float16."""
    if isinstance(tree, dict):
        return {k: _tree_map_to_f16(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [_tree_map_to_f16(v) for v in tree]
    elif isinstance(tree, mx.array):
        return tree.astype(mx.float16)
    else:
        return tree


def _tree_map_to_f32(tree):
    """Recursively convert all mx.arrays in a nested dict/list to float32."""
    if isinstance(tree, dict):
        return {k: _tree_map_to_f32(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [_tree_map_to_f32(v) for v in tree]
    elif isinstance(tree, mx.array):
        return tree.astype(mx.float32)
    else:
        return tree


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_mlx_dispatch(pipeline) -> Optional[MLXDispatchUNet]:
    """Replace pipeline's inner UNet with MLX dispatch wrapper.

    Args:
        pipeline: HunyuanPaintPipeline instance (pipeline.unet is outer UNet)

    Returns:
        MLXDispatchUNet instance, or None if setup fails
    """
    # Find checkpoint path
    ckpt_path = None
    for path in [
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--tencent--Hunyuan3D-2.1/"
            "snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/"
            "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin"
        ),
    ]:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        print("  [MLX] Checkpoint not found, falling back to PyTorch")
        return None

    print("  [MLX] Building MLX inner UNet...")
    t0 = time.time()
    mlx_model = HunyuanInnerUNet()
    print(f"  [MLX] Model built in {time.time() - t0:.1f}s")

    load_weights_from_pytorch(mlx_model, ckpt_path)

    # Precision: fp16 (default) or fp32 (HUNYUAN_MLX_FP32=1)
    # The GEGLU bug was the real quality issue, not precision. fp16 should work fine.
    use_fp32 = os.environ.get("HUNYUAN_MLX_FP32", "0") == "1"
    if use_fp32:
        print("  [MLX] Converting to float32 (HUNYUAN_MLX_FP32=1)...")
        params = mlx_model.parameters()
        params_f32 = _tree_map_to_f32(params)
        mlx_model.update(params_f32)
    else:
        print("  [MLX] Using fp16 weights (native checkpoint precision)")
    # Force materialization of all weight arrays on Metal
    mx.eval(mlx_model.parameters())  # noqa: S307 — MLX graph eval, not Python eval
    print(f"  [MLX] Model ready")

    # Replace inner UNet
    outer_unet = pipeline.unet
    original_inner = outer_unet.unet
    dispatch = MLXDispatchUNet(original_inner, mlx_model, use_fp32=use_fp32)
    outer_unet.unet = dispatch

    print(f"  [MLX] Inner UNet replaced with MLX dispatch wrapper")
    print(f"  [MLX] Step 1: PyTorch MPS, Steps 2+: MLX Metal")

    return dispatch
