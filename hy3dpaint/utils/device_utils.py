"""
Device auto-detection for cross-platform support (CUDA, MPS, CPU).

This module provides utilities to automatically select the best available
compute device, replacing hardcoded device strings throughout the codebase.
"""

import torch


def get_device() -> str:
    """Auto-detect best available compute device.

    Returns "cuda" on NVIDIA GPUs, "mps" on Apple Silicon, "cpu" otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache():
    """Clear GPU memory cache for the current backend."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
