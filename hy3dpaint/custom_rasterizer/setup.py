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

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension

# Build custom rasterizer — use CUDA when available, CPU-only otherwise (macOS/non-NVIDIA)
if torch.cuda.is_available():
    from torch.utils.cpp_extension import CUDAExtension
    custom_rasterizer_module = CUDAExtension(
        "custom_rasterizer_kernel",
        [
            "lib/custom_rasterizer_kernel/rasterizer.cpp",
            "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
            "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
        ],
    )
else:
    from torch.utils.cpp_extension import CppExtension
    custom_rasterizer_module = CppExtension(
        "custom_rasterizer_kernel",
        [
            "lib/custom_rasterizer_kernel/rasterizer.cpp",
            "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        ],
        extra_compile_args=["-Wno-c++11-narrowing"],
    )

setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_data={"custom_rasterizer": ["*.metal"]},
    package_dir={"": "."},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={"build_ext": BuildExtension},
)
