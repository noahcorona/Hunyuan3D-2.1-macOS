// Metal compute shader rasterizer — drop-in replacement for CUDA/CPU rasterizer
// Three-pass approach:
//   Pass 1 (per-face): rasterize depth only, atomic_min into 32-bit depth buffer
//   Pass 2 (per-face): compare depth, write face_id where depth matches (any face at
//                       winning depth is acceptable — same as z-fighting behavior)
//   Pass 3 (per-pixel): read face_id, compute perspective-corrected barycentric
//
// This avoids the 16-bit face_id limit of the packed 32-bit approach and works within
// Metal's constraint of no 64-bit atomics in device memory.

#include <metal_stdlib>
using namespace metal;

struct RasterParams {
    int width;
    int height;
    int num_faces;
    float occlusion_truncation;
    int use_depth_prior;
};

// Signed area * 2 for barycentric computation
inline float signed_area2(float2 a, float2 b, float2 c) {
    return (c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y);
}

inline float3 compute_barycentric_coords(float2 a, float2 b, float2 c, float2 p) {
    float beta_tri = signed_area2(a, p, c);
    float gamma_tri = signed_area2(a, b, p);
    float area = signed_area2(a, b, c);
    if (area == 0.0f) return float3(-1.0f);
    float inv = 1.0f / area;
    float beta = beta_tri * inv;
    float gamma = gamma_tri * inv;
    float alpha = 1.0f - beta - gamma;
    return float3(alpha, beta, gamma);
}

inline bool bary_in_bounds(float3 b) {
    return b.x >= 0.0f && b.x <= 1.0f &&
           b.y >= 0.0f && b.y <= 1.0f &&
           b.z >= 0.0f && b.z <= 1.0f;
}

// Clip-space to screen-space transform (matches CPU rasterizer exactly)
inline float3 clip_to_screen(float4 v, int width, int height) {
    return float3(
        (v.x / v.w * 0.5f + 0.5f) * (width - 1) + 0.5f,
        (0.5f + 0.5f * v.y / v.w) * (height - 1) + 0.5f,
        v.z / v.w * 0.49999f + 0.5f
    );
}

// Pass 1: Depth-only rasterization — one thread per face
// Writes min depth per pixel using 32-bit atomic min (full depth precision)
kernel void rasterize_depth(
    device const float4* vertices  [[buffer(0)]],
    device const int*    faces     [[buffer(1)]],
    device const float*  depth_prior [[buffer(2)]],
    device atomic_uint*  depth_buf  [[buffer(3)]],
    constant RasterParams& params  [[buffer(4)]],
    uint face_id [[thread_position_in_grid]])
{
    if (face_id >= uint(params.num_faces)) return;

    int width  = params.width;
    int height = params.height;

    float4 v0 = vertices[faces[face_id * 3]];
    float4 v1 = vertices[faces[face_id * 3 + 1]];
    float4 v2 = vertices[faces[face_id * 3 + 2]];

    float3 s0 = clip_to_screen(v0, width, height);
    float3 s1 = clip_to_screen(v1, width, height);
    float3 s2 = clip_to_screen(v2, width, height);

    int x_min = max(0, int(floor(min(s0.x, min(s1.x, s2.x)))));
    int x_max = min(width - 1, int(floor(max(s0.x, max(s1.x, s2.x)))) + 1);
    int y_min = max(0, int(floor(min(s0.y, min(s1.y, s2.y)))));
    int y_max = min(height - 1, int(floor(max(s0.y, max(s1.y, s2.y)))) + 1);

    for (int px = x_min; px <= x_max; ++px) {
        for (int py = y_min; py <= y_max; ++py) {
            float2 p = float2(float(px) + 0.5f, float(py) + 0.5f);
            float3 bary = compute_barycentric_coords(s0.xy, s1.xy, s2.xy, p);

            if (bary_in_bounds(bary)) {
                int pixel = py * width + px;
                float depth = bary.x * s0.z + bary.y * s1.z + bary.z * s2.z;

                if (params.use_depth_prior) {
                    float depth_thres = depth_prior[pixel] * 0.49999f + 0.5f
                                      + params.occlusion_truncation;
                    if (depth < depth_thres) continue;
                }

                uint z_u32 = uint(clamp(depth * 4294967295.0f, 0.0f, 4294967295.0f));
                atomic_fetch_min_explicit(&depth_buf[pixel], z_u32, memory_order_relaxed);
            }
        }
    }
}

// Pass 2: Face assignment — one thread per face
// For each pixel where this face's depth matches the z-buffer, write face_id.
// Race: multiple faces at same depth may write — any winner is acceptable.
kernel void assign_faces(
    device const float4* vertices  [[buffer(0)]],
    device const int*    faces     [[buffer(1)]],
    device const float*  depth_prior [[buffer(2)]],
    device const uint*   depth_buf  [[buffer(3)]],
    device int*          face_buf   [[buffer(4)]],
    constant RasterParams& params  [[buffer(5)]],
    uint face_id [[thread_position_in_grid]])
{
    if (face_id >= uint(params.num_faces)) return;

    int width  = params.width;
    int height = params.height;

    float4 v0 = vertices[faces[face_id * 3]];
    float4 v1 = vertices[faces[face_id * 3 + 1]];
    float4 v2 = vertices[faces[face_id * 3 + 2]];

    float3 s0 = clip_to_screen(v0, width, height);
    float3 s1 = clip_to_screen(v1, width, height);
    float3 s2 = clip_to_screen(v2, width, height);

    int x_min = max(0, int(floor(min(s0.x, min(s1.x, s2.x)))));
    int x_max = min(width - 1, int(floor(max(s0.x, max(s1.x, s2.x)))) + 1);
    int y_min = max(0, int(floor(min(s0.y, min(s1.y, s2.y)))));
    int y_max = min(height - 1, int(floor(max(s0.y, max(s1.y, s2.y)))) + 1);

    for (int px = x_min; px <= x_max; ++px) {
        for (int py = y_min; py <= y_max; ++py) {
            float2 p = float2(float(px) + 0.5f, float(py) + 0.5f);
            float3 bary = compute_barycentric_coords(s0.xy, s1.xy, s2.xy, p);

            if (bary_in_bounds(bary)) {
                int pixel = py * width + px;
                float depth = bary.x * s0.z + bary.y * s1.z + bary.z * s2.z;

                if (params.use_depth_prior) {
                    float depth_thres = depth_prior[pixel] * 0.49999f + 0.5f
                                      + params.occlusion_truncation;
                    if (depth < depth_thres) continue;
                }

                uint z_u32 = uint(clamp(depth * 4294967295.0f, 0.0f, 4294967295.0f));
                if (depth_buf[pixel] == z_u32) {
                    face_buf[pixel] = int(face_id + 1);
                }
            }
        }
    }
}

// Pass 3: Per-pixel barycentric with perspective correction — one thread per pixel
kernel void compute_barycentric(
    device const float4* vertices      [[buffer(0)]],
    device const int*    faces         [[buffer(1)]],
    device const int*    face_buf      [[buffer(2)]],
    device int*          findices      [[buffer(3)]],
    device float*        barycentric_map [[buffer(4)]],
    constant RasterParams& params      [[buffer(5)]],
    uint pixel_id [[thread_position_in_grid]])
{
    int width  = params.width;
    int height = params.height;
    if (pixel_id >= uint(width * height)) return;

    int face_1based = face_buf[pixel_id];

    if (face_1based == 0) {
        findices[pixel_id] = 0;
        barycentric_map[pixel_id * 3]     = 0.0f;
        barycentric_map[pixel_id * 3 + 1] = 0.0f;
        barycentric_map[pixel_id * 3 + 2] = 0.0f;
        return;
    }

    findices[pixel_id] = face_1based;
    int f = face_1based - 1;

    float2 p = float2(float(int(pixel_id) % width) + 0.5f,
                       float(int(pixel_id) / width) + 0.5f);

    float4 v0 = vertices[faces[f * 3]];
    float4 v1 = vertices[faces[f * 3 + 1]];
    float4 v2 = vertices[faces[f * 3 + 2]];

    float2 s0 = float2(
        (v0.x / v0.w * 0.5f + 0.5f) * (width - 1) + 0.5f,
        (0.5f + 0.5f * v0.y / v0.w) * (height - 1) + 0.5f);
    float2 s1 = float2(
        (v1.x / v1.w * 0.5f + 0.5f) * (width - 1) + 0.5f,
        (0.5f + 0.5f * v1.y / v1.w) * (height - 1) + 0.5f);
    float2 s2 = float2(
        (v2.x / v2.w * 0.5f + 0.5f) * (width - 1) + 0.5f,
        (0.5f + 0.5f * v2.y / v2.w) * (height - 1) + 0.5f);

    float3 bary = compute_barycentric_coords(s0, s1, s2, p);

    // Perspective correction: divide by w, then normalize
    bary.x /= v0.w;
    bary.y /= v1.w;
    bary.z /= v2.w;
    float w = 1.0f / (bary.x + bary.y + bary.z);
    bary *= w;

    barycentric_map[pixel_id * 3]     = bary.x;
    barycentric_map[pixel_id * 3 + 1] = bary.y;
    barycentric_map[pixel_id * 3 + 2] = bary.z;
}
