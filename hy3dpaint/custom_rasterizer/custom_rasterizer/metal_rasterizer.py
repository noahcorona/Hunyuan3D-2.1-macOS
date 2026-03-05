"""
Metal GPU rasterizer — uses Apple's Metal compute shaders for triangle rasterization.
Drop-in replacement for the C++ custom_rasterizer_kernel.rasterize_image function.

Three-pass approach (avoids Metal's lack of 64-bit atomics):
  Pass 1: depth-only rasterization with 32-bit atomic min (full precision)
  Pass 2: face assignment — write face_id where depth matches z-buffer
  Pass 3: per-pixel barycentric computation with perspective correction

Requires: pyobjc-framework-Metal (pip install pyobjc-framework-Metal)
Works on: Apple Silicon Macs (M1/M2/M3/M4) with macOS 12+
"""

import os
import struct
import numpy as np
import torch

_metal_state = None  # Lazy-init singleton


class MetalRasterizer:
    """Manages Metal device, pipelines, and command queue for rasterization."""

    def __init__(self):
        import Metal
        import objc

        self.Metal = Metal

        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device found")

        # Compile shader from source at runtime
        shader_path = os.path.join(os.path.dirname(__file__), "rasterizer.metal")
        with open(shader_path, "r") as f:
            shader_source = f.read()

        options = Metal.MTLCompileOptions.new()
        library, error = self.device.newLibraryWithSource_options_error_(
            shader_source, options, None
        )
        if error:
            raise RuntimeError(f"Metal shader compilation failed: {error}")

        # Create compute pipelines for all 3 passes
        depth_fn = library.newFunctionWithName_("rasterize_depth")
        assign_fn = library.newFunctionWithName_("assign_faces")
        bary_fn = library.newFunctionWithName_("compute_barycentric")
        if depth_fn is None or assign_fn is None or bary_fn is None:
            raise RuntimeError("Metal function not found in compiled library")

        self.depth_pipeline, error = (
            self.device.newComputePipelineStateWithFunction_error_(depth_fn, None)
        )
        if error:
            raise RuntimeError(f"Metal pipeline creation failed: {error}")

        self.assign_pipeline, error = (
            self.device.newComputePipelineStateWithFunction_error_(assign_fn, None)
        )
        if error:
            raise RuntimeError(f"Metal pipeline creation failed: {error}")

        self.barycentric_pipeline, error = (
            self.device.newComputePipelineStateWithFunction_error_(bary_fn, None)
        )
        if error:
            raise RuntimeError(f"Metal pipeline creation failed: {error}")

        self.command_queue = self.device.newCommandQueue()
        self._max_tg_depth = self.depth_pipeline.maxTotalThreadsPerThreadgroup()
        self._max_tg_assign = self.assign_pipeline.maxTotalThreadsPerThreadgroup()
        self._max_tg_bary = self.barycentric_pipeline.maxTotalThreadsPerThreadgroup()

    def _buf_from_numpy(self, arr):
        """Create a shared Metal buffer from a contiguous numpy array."""
        arr = np.ascontiguousarray(arr)
        data = arr.tobytes()
        return self.device.newBufferWithBytes_length_options_(
            data, len(data), self.Metal.MTLResourceStorageModeShared
        )

    def _empty_buf(self, nbytes):
        """Create an empty shared Metal buffer."""
        return self.device.newBufferWithLength_options_(
            nbytes, self.Metal.MTLResourceStorageModeShared
        )

    def rasterize_image(self, V_np, F_np, D_np, width, height,
                        occlusion_truncation, use_depth_prior):
        """
        Metal GPU rasterization.

        Args:
            V_np: (N, 4) float32 — vertices in clip space (x, y, z, w)
            F_np: (M, 3) int32 — triangle face indices
            D_np: (H, W) float32 — depth prior (or empty array)
            width, height: image dimensions
            occlusion_truncation: float
            use_depth_prior: int (0 or 1)

        Returns:
            (findices, barycentric) as numpy arrays:
                findices: (H, W) int32 — 1-based face index per pixel (0 = background)
                barycentric: (H, W, 3) float32 — perspective-corrected barycentric coords
        """
        num_faces = F_np.shape[0]
        total_pixels = width * height

        # Params struct (must match Metal struct layout: 5 x 4 bytes)
        params_data = struct.pack(
            "iiifi", width, height, num_faces,
            occlusion_truncation, use_depth_prior
        )

        # Create Metal buffers
        V_buf = self._buf_from_numpy(V_np.astype(np.float32))
        F_buf = self._buf_from_numpy(F_np.astype(np.int32).reshape(-1))

        if use_depth_prior and D_np.size > 0:
            D_buf = self._buf_from_numpy(D_np.astype(np.float32))
        else:
            D_buf = self._empty_buf(4)  # Dummy buffer (never read)

        # Depth buffer: init to 0xFFFFFFFF (max uint32 = farthest)
        depth_np = np.full(total_pixels, 0xFFFFFFFF, dtype=np.uint32)
        depth_buf = self._buf_from_numpy(depth_np)

        # Face ID buffer: init to 0 (background)
        face_np = np.zeros(total_pixels, dtype=np.int32)
        face_buf = self._buf_from_numpy(face_np)

        params_buf = self.device.newBufferWithBytes_length_options_(
            params_data, len(params_data), self.Metal.MTLResourceStorageModeShared
        )

        # Output buffers
        findices_buf = self._empty_buf(total_pixels * 4)
        bary_buf = self._empty_buf(total_pixels * 3 * 4)

        # --- Pass 1: depth-only rasterization ---
        cmd1 = self.command_queue.commandBuffer()
        enc1 = cmd1.computeCommandEncoder()
        enc1.setComputePipelineState_(self.depth_pipeline)
        enc1.setBuffer_offset_atIndex_(V_buf, 0, 0)
        enc1.setBuffer_offset_atIndex_(F_buf, 0, 1)
        enc1.setBuffer_offset_atIndex_(D_buf, 0, 2)
        enc1.setBuffer_offset_atIndex_(depth_buf, 0, 3)
        enc1.setBuffer_offset_atIndex_(params_buf, 0, 4)
        tg1 = min(256, self._max_tg_depth)
        enc1.dispatchThreads_threadsPerThreadgroup_(
            self.Metal.MTLSizeMake(num_faces, 1, 1),
            self.Metal.MTLSizeMake(tg1, 1, 1),
        )
        enc1.endEncoding()
        cmd1.commit()
        cmd1.waitUntilCompleted()
        if cmd1.error():
            raise RuntimeError(f"Metal pass 1 error: {cmd1.error()}")

        # --- Pass 2: face assignment ---
        cmd2 = self.command_queue.commandBuffer()
        enc2 = cmd2.computeCommandEncoder()
        enc2.setComputePipelineState_(self.assign_pipeline)
        enc2.setBuffer_offset_atIndex_(V_buf, 0, 0)
        enc2.setBuffer_offset_atIndex_(F_buf, 0, 1)
        enc2.setBuffer_offset_atIndex_(D_buf, 0, 2)
        enc2.setBuffer_offset_atIndex_(depth_buf, 0, 3)  # read-only now
        enc2.setBuffer_offset_atIndex_(face_buf, 0, 4)
        enc2.setBuffer_offset_atIndex_(params_buf, 0, 5)
        tg2 = min(256, self._max_tg_assign)
        enc2.dispatchThreads_threadsPerThreadgroup_(
            self.Metal.MTLSizeMake(num_faces, 1, 1),
            self.Metal.MTLSizeMake(tg2, 1, 1),
        )
        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()
        if cmd2.error():
            raise RuntimeError(f"Metal pass 2 error: {cmd2.error()}")

        # --- Pass 3: per-pixel barycentric ---
        cmd3 = self.command_queue.commandBuffer()
        enc3 = cmd3.computeCommandEncoder()
        enc3.setComputePipelineState_(self.barycentric_pipeline)
        enc3.setBuffer_offset_atIndex_(V_buf, 0, 0)
        enc3.setBuffer_offset_atIndex_(F_buf, 0, 1)
        enc3.setBuffer_offset_atIndex_(face_buf, 0, 2)  # read face IDs
        enc3.setBuffer_offset_atIndex_(findices_buf, 0, 3)
        enc3.setBuffer_offset_atIndex_(bary_buf, 0, 4)
        enc3.setBuffer_offset_atIndex_(params_buf, 0, 5)
        tg3 = min(256, self._max_tg_bary)
        enc3.dispatchThreads_threadsPerThreadgroup_(
            self.Metal.MTLSizeMake(total_pixels, 1, 1),
            self.Metal.MTLSizeMake(tg3, 1, 1),
        )
        enc3.endEncoding()
        cmd3.commit()
        cmd3.waitUntilCompleted()
        if cmd3.error():
            raise RuntimeError(f"Metal pass 3 error: {cmd3.error()}")

        # Read results back from GPU buffers
        fi_ptr = findices_buf.contents()
        bc_ptr = bary_buf.contents()

        fi_bytes = fi_ptr.as_buffer(total_pixels * 4)
        bc_bytes = bc_ptr.as_buffer(total_pixels * 3 * 4)

        findices = np.frombuffer(bytes(fi_bytes), dtype=np.int32).reshape(height, width)
        barycentric = np.frombuffer(bytes(bc_bytes), dtype=np.float32).reshape(
            height, width, 3
        )

        return findices.copy(), barycentric.copy()


def _get_metal():
    """Lazy-init the Metal rasterizer singleton."""
    global _metal_state
    if _metal_state is None:
        _metal_state = MetalRasterizer()
    return _metal_state


def metal_available():
    """Check if Metal rasterization is available."""
    try:
        _get_metal()
        return True
    except Exception:
        return False


def rasterize_image_metal(V, F, D, width, height, occlusion_truncation, use_depth_prior):
    """
    PyTorch-compatible wrapper. Accepts and returns torch tensors.

    Args:
        V: (N, 4) float tensor — vertices in clip space
        F: (M, 3) int tensor — face indices
        D: depth prior tensor (or empty)
        width, height: int
        occlusion_truncation: float
        use_depth_prior: int

    Returns:
        (findices, barycentric) as torch tensors on CPU
    """
    metal = _get_metal()

    V_np = V.detach().cpu().float().numpy()
    F_np = F.detach().cpu().int().numpy()
    D_np = D.detach().cpu().float().numpy() if D.numel() > 0 else np.zeros(0, dtype=np.float32)

    fi_np, bc_np = metal.rasterize_image(
        V_np, F_np, D_np, width, height, occlusion_truncation, use_depth_prior
    )

    return torch.from_numpy(fi_np), torch.from_numpy(bc_np)
