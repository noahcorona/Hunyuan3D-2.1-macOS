# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import custom_rasterizer_kernel
import torch

# Try Metal GPU rasterizer first (Apple Silicon)
try:
    from .metal_rasterizer import rasterize_image_metal, metal_available
    _use_metal = metal_available()
except Exception:
    _use_metal = False


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    assert pos.device == tri.device
    target_device = pos.device
    if _use_metal:
        findices, barycentric = rasterize_image_metal(
            pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
        )
        findices = findices.to(target_device)
        barycentric = barycentric.to(target_device)
    else:
        findices, barycentric = custom_rasterizer_kernel.rasterize_image(
            pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
        )

    # Sanitize NaN/inf from Metal rasterizer output
    if _use_metal and (barycentric.isnan().any() or barycentric.isinf().any()):
        barycentric = torch.nan_to_num(barycentric, nan=0.0, posinf=0.0, neginf=0.0)
        bad_pixels = (barycentric == 0).all(dim=-1)
        findices[bad_pixels] = 0

    return findices, barycentric


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
