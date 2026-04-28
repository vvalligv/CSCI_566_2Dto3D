"""
compute_image_metrics.py — Render predicted + GT meshes from the same viewpoint,
compute PSNR and LPIPS, and save side-by-side comparison images.

Requires: pip install lpips scikit-image

Usage:
  python compute_image_metrics.py --base-dir /path/to/project
"""

import argparse
import json
import os
import sys
import numpy as np
import trimesh
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr


def parse_args():
    p = argparse.ArgumentParser(description="Compute PSNR/LPIPS and render comparison images")
    p.add_argument("--base-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                    help="Project root")
    p.add_argument("--split-json", type=str, default=None)
    p.add_argument("--inf-manifest", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None,
                    help="Where to save comparison images (default: <base-dir>/comparison_images)")
    p.add_argument("--img-size", type=int, default=256, help="Render resolution for comparison")
    return p.parse_args()


def load_and_normalize(path):
    scene_or_mesh = trimesh.load(path, force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        scene_or_mesh = scene_or_mesh.dump(concatenate=True)
    mesh = trimesh.Trimesh(vertices=scene_or_mesh.vertices,
                           faces=scene_or_mesh.faces, process=False)
    mesh.apply_translation(-mesh.centroid)
    ext = max(mesh.extents)
    if ext > 1e-8:
        mesh.apply_scale(1.0 / ext)
    return mesh


def render_mesh_raycasting(mesh, height, width, cam_dist=1.9, fov_deg=40.0):
    """Render a mesh from front view using ray casting. Returns RGB [H,W,3] uint8."""
    fov_rad = np.radians(fov_deg)
    half_h = cam_dist * np.tan(fov_rad / 2)

    ys = np.linspace(half_h, -half_h, height)
    xs = np.linspace(-half_h, half_h, width)
    xx, yy = np.meshgrid(xs, ys)

    origins = np.zeros((height * width, 3), dtype=np.float64)
    origins[:, 0] = xx.ravel()
    origins[:, 1] = yy.ravel()
    origins[:, 2] = cam_dist

    directions = np.zeros((height * width, 3), dtype=np.float64)
    directions[:, 2] = -1.0

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=False)

    rgb = np.ones((height * width, 3), dtype=np.float32) * 255
    if len(locations) > 0:
        face_normals = mesh.face_normals[index_tri]
        light_dir = np.array([0.0, 0.0, 1.0])
        diffuse = np.abs(face_normals @ light_dir).astype(np.float32)
        diffuse = np.clip(0.3 + 0.7 * diffuse, 0.0, 1.0)
        rgb[index_ray] = (diffuse[:, None] * 255).astype(np.float32)

    return rgb.reshape(height, width, 3).astype(np.uint8)


def main():
    import lpips

    args = parse_args()
    base = args.base_dir
    data_dir = os.path.join(base, "project_scripts/project_scripts")
    H = W = args.img_size

    split_json = args.split_json or os.path.join(data_dir, "dataset_split.json")
    inf_manifest = args.inf_manifest or os.path.join(data_dir, "inference_manifest_backface_v2.json")
    output_dir = args.output_dir or os.path.join(base, "comparison_images")
    output_json = os.path.join(data_dir, "image_metrics_v2.json")

    os.makedirs(output_dir, exist_ok=True)

    lpips_fn = lpips.LPIPS(net="alex")

    with open(split_json) as f:
        split = json.load(f)
    with open(inf_manifest) as f:
        inf_data = json.load(f)

    gt_paths = {}
    for subset in ["hard_set", "easy_set"]:
        for obj in split[subset]:
            gt_paths[obj["uid"]] = obj["glb_path"]

    results = {"hard": [], "easy": []}
    all_psnr = {"hard": [], "easy": []}
    all_lpips = {"hard": [], "easy": []}

    for uid, info in inf_data.items():
        label = info["label"]
        pred_path = info["mesh_path"]
        gt_path = gt_paths.get(uid)

        if gt_path is None or not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"  SKIP {uid[:8]}: missing files")
            continue

        print(f"  Rendering {uid[:8]}... ({label})")
        try:
            pred_mesh = load_and_normalize(pred_path)
            gt_mesh = load_and_normalize(gt_path)

            pred_img = render_mesh_raycasting(pred_mesh, H, W)
            gt_img = render_mesh_raycasting(gt_mesh, H, W)

            psnr_val = compute_psnr(gt_img, pred_img, data_range=255)

            pred_t = torch.from_numpy(pred_img).permute(2, 0, 1).float() / 127.5 - 1.0
            gt_t = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 127.5 - 1.0
            with torch.no_grad():
                lpips_val = lpips_fn(pred_t.unsqueeze(0), gt_t.unsqueeze(0)).item()

            results[label].append({"uid": uid, "psnr": psnr_val, "lpips": lpips_val})
            all_psnr[label].append(psnr_val)
            all_lpips[label].append(lpips_val)
            print(f"    PSNR={psnr_val:.2f}  LPIPS={lpips_val:.4f}")

            combined = np.zeros((H, W * 2 + 10, 3), dtype=np.uint8)
            combined[:, :W] = gt_img
            combined[:, W + 10:] = pred_img
            combined[:, W:W + 10] = 128
            Image.fromarray(combined).save(os.path.join(output_dir, f"{uid[:8]}_{label}.png"))

        except Exception as e:
            print(f"  ERROR {uid[:8]}: {e}")

    print(f"\n{'='*65}")
    print("IMAGE METRICS: PSNR (higher=better) and LPIPS (lower=better)")
    print(f"{'='*65}")
    print(f"{'':20} {'PSNR':>12} {'LPIPS':>12} {'N':>6}")
    print(f"{'-'*65}")
    for label in ["hard", "easy"]:
        if all_psnr[label]:
            print(f"{label.capitalize():20} "
                  f"{np.mean(all_psnr[label]):>12.2f} "
                  f"{np.mean(all_lpips[label]):>12.4f} "
                  f"{len(all_psnr[label]):>6}")
    print(f"{'='*65}")
    print(f"\nComparison images saved to {output_dir}/  (left=GT, right=predicted)")

    with open(output_json, "w") as f:
        json.dump({
            "per_object": results,
            "summary": {
                l: {"mean_psnr": float(np.mean(all_psnr[l])) if all_psnr[l] else None,
                    "mean_lpips": float(np.mean(all_lpips[l])) if all_lpips[l] else None,
                    "n": len(all_psnr[l])}
                for l in ["hard", "easy"]
            }
        }, f, indent=2)
    print(f"Saved to {output_json}")


if __name__ == "__main__":
    main()
