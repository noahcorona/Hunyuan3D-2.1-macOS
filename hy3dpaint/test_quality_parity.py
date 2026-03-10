"""End-to-end quality parity test: MLX backend vs PyTorch MPS backend.

Runs the full texture paint pipeline on the same mesh+image with both backends,
then compares the output texture maps and mesh geometry.

Usage:
    python test_quality_parity.py --mesh path/to/mesh.glb --image path/to/image.png
    python test_quality_parity.py  # uses default example assets

The test produces:
  - PSNR and mean absolute error for each texture map (albedo, metallic, roughness)
  - Chamfer distance between the two output meshes
  - Side-by-side comparison images saved to the output directory

Note: This requires both backends to be available (MLX + PyTorch MPS).
      Each run takes ~3-6 min on Apple Silicon, so total ~6-12 min.
"""

import argparse
import os
import sys
import shutil
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hy3dshape'))


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_mae(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute Mean Absolute Error between two images (0-255 scale)."""
    return np.mean(np.abs(img_a.astype(np.float64) - img_b.astype(np.float64)))


def save_diff_image(img_a: np.ndarray, img_b: np.ndarray, path: str):
    """Save an amplified absolute difference image."""
    diff = np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))
    # Amplify for visibility (10x)
    diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(path)


def compare_texture_maps(dir_a: str, dir_b: str, output_dir: str) -> dict:
    """Compare texture maps between two pipeline output directories.

    Returns dict of {map_name: {psnr, mae, max_diff}} for each texture found.
    """
    texture_suffixes = [
        ("albedo", ".jpg"),
        ("metallic", "_metallic.jpg"),
        ("roughness", "_roughness.jpg"),
    ]

    results = {}
    for name, suffix in texture_suffixes:
        # Find the texture file in each directory
        file_a = os.path.join(dir_a, f"textured_mesh{suffix}")
        file_b = os.path.join(dir_b, f"textured_mesh{suffix}")

        if not os.path.exists(file_a):
            print(f"  {name}: not found in PyTorch output ({file_a})")
            continue
        if not os.path.exists(file_b):
            print(f"  {name}: not found in MLX output ({file_b})")
            continue

        img_a = np.array(Image.open(file_a).convert("RGB"))
        img_b = np.array(Image.open(file_b).convert("RGB"))

        if img_a.shape != img_b.shape:
            print(f"  {name}: shape mismatch {img_a.shape} vs {img_b.shape}")
            results[name] = {"error": "shape_mismatch"}
            continue

        psnr = compute_psnr(img_a, img_b)
        mae = compute_mae(img_a, img_b)
        max_diff = float(np.max(np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))))

        results[name] = {
            "psnr_db": round(psnr, 2),
            "mae_0_255": round(mae, 2),
            "max_diff": max_diff,
            "resolution": f"{img_a.shape[1]}x{img_a.shape[0]}",
        }

        # Save diff visualization
        diff_path = os.path.join(output_dir, f"diff_{name}.png")
        save_diff_image(img_a, img_b, diff_path)

        # Save side-by-side
        side_by_side = np.concatenate([img_a, img_b], axis=1)
        sbs_path = os.path.join(output_dir, f"sidebyside_{name}.png")
        Image.fromarray(side_by_side).save(sbs_path)

    return results


def compare_meshes(obj_a: str, obj_b: str) -> dict:
    """Compare two OBJ meshes using Chamfer distance."""
    try:
        import trimesh
    except ImportError:
        return {"error": "trimesh not installed"}

    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'hy3dshape', 'tools', 'evaluation'
    ))
    from chamfer_distance import chamfer_distance_from_meshes

    mesh_a = trimesh.load(obj_a, force='mesh', process=False)
    mesh_b = trimesh.load(obj_b, force='mesh', process=False)

    metrics = chamfer_distance_from_meshes(
        np.asarray(mesh_a.vertices), np.asarray(mesh_a.faces),
        np.asarray(mesh_b.vertices), np.asarray(mesh_b.faces),
        n_samples=50000,
    )

    return {
        "chamfer_l2": round(metrics["cd_l2"], 6),
        "a_to_b_l2": round(metrics["A_to_B_l2"], 6),
        "b_to_a_l2": round(metrics["B_to_A_l2"], 6),
        "mesh_a_faces": mesh_a.faces.shape[0],
        "mesh_b_faces": mesh_b.faces.shape[0],
    }


def run_paint(mesh_path: str, image_path: str, output_dir: str,
              use_mlx: bool, num_inference_steps: int = 15) -> str:
    """Run the paint pipeline with one backend. Returns path to output OBJ."""
    import subprocess

    script = f"""
import os, sys
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
sys.path.insert(0, os.path.join({os.path.dirname(os.path.abspath(__file__))!r}, '..', 'hy3dshape'))

if not {use_mlx}:
    os.environ["HUNYUAN_DISABLE_MLX"] = "1"

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

cfg = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
cfg.realesrgan_ckpt_path = os.path.join({os.path.dirname(os.path.abspath(__file__))!r}, "ckpt", "RealESRGAN_x4plus.pth")
cfg.multiview_cfg_path = os.path.join({os.path.dirname(os.path.abspath(__file__))!r}, "cfgs", "hunyuan-paint-pbr.yaml")
cfg.custom_pipeline = os.path.join({os.path.dirname(os.path.abspath(__file__))!r}, "hunyuanpaintpbr")

pipeline = Hunyuan3DPaintPipeline(cfg)

output_path = os.path.join({output_dir!r}, "textured_mesh.obj")
result = pipeline(
    mesh_path={mesh_path!r},
    image_path={image_path!r},
    output_mesh_path=output_path,
    save_glb=False,
    num_inference_steps={num_inference_steps},
)
print(f"OUTPUT_PATH={{result}}")
"""

    env = os.environ.copy()
    if not use_mlx:
        env["HUNYUAN_DISABLE_MLX"] = "1"

    backend_name = "MLX" if use_mlx else "PyTorch"
    print(f"\n{'='*60}")
    print(f"  Running paint pipeline ({backend_name})")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min max
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED ({backend_name}):")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        raise RuntimeError(f"{backend_name} pipeline failed")

    # Extract output path from stdout
    for line in result.stdout.splitlines():
        if line.startswith("OUTPUT_PATH="):
            output_path = line.split("=", 1)[1]
            break
    else:
        output_path = os.path.join(output_dir, "textured_mesh.obj")

    print(f"  {backend_name} completed in {elapsed:.1f}s")
    print(f"  Output: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Quality parity test: MLX vs PyTorch texture paint")
    parser.add_argument("--mesh", type=str, default=None,
                        help="Path to input mesh (GLB or OBJ)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (PNG)")
    parser.add_argument("--output-dir", type=str, default="/tmp/quality_parity_test",
                        help="Directory for test outputs")
    parser.add_argument("--steps", type=int, default=15,
                        help="Number of diffusion inference steps")
    parser.add_argument("--skip-pytorch", action="store_true",
                        help="Skip PyTorch run (use existing output in output-dir/pytorch/)")
    parser.add_argument("--skip-mlx", action="store_true",
                        help="Skip MLX run (use existing output in output-dir/mlx/)")
    args = parser.parse_args()

    # Default to first example image if none provided
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    if args.image is None:
        args.image = os.path.join(repo_root, "assets", "example_images", "004.png")
        print(f"Using default image: {args.image}")

    if args.mesh is None:
        print("ERROR: --mesh is required (path to a shape pipeline output GLB/OBJ)")
        print("  Generate one first with the shape pipeline, e.g.:")
        print("    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline")
        print("    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')")
        print("    mesh = pipeline(image='assets/example_images/004.png')[0]")
        sys.exit(1)

    # Validate inputs
    if not os.path.exists(args.mesh):
        print(f"ERROR: mesh not found: {args.mesh}")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)

    # Setup output dirs
    pt_dir = os.path.join(args.output_dir, "pytorch")
    mlx_dir = os.path.join(args.output_dir, "mlx")
    comparison_dir = os.path.join(args.output_dir, "comparison")
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(mlx_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # Copy input mesh to both dirs (pipeline needs it writable)
    for d in [pt_dir, mlx_dir]:
        mesh_dest = os.path.join(d, os.path.basename(args.mesh))
        if not os.path.exists(mesh_dest):
            shutil.copy2(args.mesh, mesh_dest)

    print(f"\nQuality Parity Test")
    print(f"  Input mesh:  {args.mesh}")
    print(f"  Input image: {args.image}")
    print(f"  Steps:       {args.steps}")
    print(f"  Output dir:  {args.output_dir}")

    # Run both backends
    timings = {}

    if not args.skip_pytorch:
        t0 = time.time()
        pt_obj = run_paint(args.mesh, args.image, pt_dir, use_mlx=False,
                           num_inference_steps=args.steps)
        timings["pytorch_s"] = round(time.time() - t0, 1)
    else:
        pt_obj = os.path.join(pt_dir, "textured_mesh.obj")
        print(f"\nSkipping PyTorch run, using existing: {pt_obj}")

    if not args.skip_mlx:
        t0 = time.time()
        mlx_obj = run_paint(args.mesh, args.image, mlx_dir, use_mlx=True,
                            num_inference_steps=args.steps)
        timings["mlx_s"] = round(time.time() - t0, 1)
    else:
        mlx_obj = os.path.join(mlx_dir, "textured_mesh.obj")
        print(f"\nSkipping MLX run, using existing: {mlx_obj}")

    # Compare results
    print(f"\n{'='*60}")
    print(f"  Comparing outputs")
    print(f"{'='*60}")

    # Texture map comparison
    print("\n--- Texture Maps ---")
    tex_results = compare_texture_maps(pt_dir, mlx_dir, comparison_dir)
    for name, metrics in tex_results.items():
        if "error" in metrics:
            print(f"  {name}: {metrics['error']}")
        else:
            print(f"  {name}:")
            print(f"    PSNR:     {metrics['psnr_db']} dB")
            print(f"    MAE:      {metrics['mae_0_255']}/255")
            print(f"    Max diff: {metrics['max_diff']}/255")
            print(f"    Size:     {metrics['resolution']}")

    # Mesh comparison
    print("\n--- Mesh Geometry ---")
    if os.path.exists(pt_obj) and os.path.exists(mlx_obj):
        mesh_results = compare_meshes(pt_obj, mlx_obj)
        if "error" in mesh_results:
            print(f"  {mesh_results['error']}")
        else:
            print(f"  Chamfer L2:  {mesh_results['chamfer_l2']}")
            print(f"  A->B L2:     {mesh_results['a_to_b_l2']}")
            print(f"  B->A L2:     {mesh_results['b_to_a_l2']}")
            print(f"  Faces (PT):  {mesh_results['mesh_a_faces']}")
            print(f"  Faces (MLX): {mesh_results['mesh_b_faces']}")
    else:
        mesh_results = {"error": "missing output meshes"}
        print("  Skipped (missing output meshes)")

    # Timing comparison
    if timings:
        print("\n--- Timing ---")
        for k, v in timings.items():
            print(f"  {k}: {v}s")
        if "pytorch_s" in timings and "mlx_s" in timings:
            speedup = timings["pytorch_s"] / timings["mlx_s"]
            print(f"  Speedup: {speedup:.2f}x")

    # Summary verdict
    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")

    all_psnr = [v["psnr_db"] for v in tex_results.values() if "psnr_db" in v]
    if all_psnr:
        min_psnr = min(all_psnr)
        if min_psnr > 30:
            print(f"  PASS - Texture quality parity confirmed (min PSNR: {min_psnr} dB)")
            print(f"         PSNR > 30 dB indicates visually indistinguishable output")
        elif min_psnr > 20:
            print(f"  ACCEPTABLE - Minor texture differences (min PSNR: {min_psnr} dB)")
            print(f"         PSNR 20-30 dB: minor visible differences, typical for fp16 backends")
        else:
            print(f"  INVESTIGATE - Significant texture differences (min PSNR: {min_psnr} dB)")
            print(f"         PSNR < 20 dB: visible quality difference, check diff images")
    else:
        print("  INCOMPLETE - Could not compare texture maps")

    print(f"\n  Diff images saved to: {comparison_dir}")
    print(f"  PyTorch output: {pt_dir}")
    print(f"  MLX output:     {mlx_dir}")


if __name__ == "__main__":
    main()
