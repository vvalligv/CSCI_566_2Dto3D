"""
compute_metrics.py
Scores each predicted mesh (one per input view) against the GT GLB.

Metrics per object:
  3D:    Chamfer Distance, F-score
  Image: PSNR, LPIPS       (color/vertex renders)
         PSNR-N, LPIPS-N   (surface normal renders)

Output: metrics_baseline.json
"""
import json
import os
import numpy as np
import trimesh
import torch
import lpips
from scipy.spatial import cKDTree
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
import pyrender

os.environ["PYOPENGL_PLATFORM"] = "egl"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SPLIT_JSON   = os.path.join(SCRIPT_DIR, "dataset_split.json")
INF_MANIFEST = os.path.join(SCRIPT_DIR, "inference_manifest.json")
OUTPUT_JSON  = os.path.join(SCRIPT_DIR, "metrics_baseline.json")

N_SAMPLES  = 10000
THRESHOLD  = 0.05
RESOLUTION = 512
N_VIEWS    = 8
CAM_DIST   = 1.8
YFOV       = np.pi / 3.0
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Camera poses ──────────────────────────────────────────────────────────────

def get_camera_poses(n_views=N_VIEWS, cam_dist=CAM_DIST, elevation_deg=20.0):
    poses = []
    elevation = np.radians(elevation_deg)
    for i in range(n_views):
        azimuth = 2 * np.pi * i / n_views
        x = cam_dist * np.cos(elevation) * np.sin(azimuth)
        y = cam_dist * np.sin(elevation)
        z = cam_dist * np.cos(elevation) * np.cos(azimuth)
        cam_pos = np.array([x, y, z])

        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        pose = np.eye(4)
        pose[:3, 0] =  right
        pose[:3, 1] =  up
        pose[:3, 2] = -forward
        pose[:3, 3] =  cam_pos
        poses.append(pose.astype(np.float32))
    return poses


CAMERA_POSES = get_camera_poses()


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_and_normalize(path):
    """Load mesh preserving vertex colors/materials, normalize to unit size."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))
    return mesh


def sample_surface(mesh, n=N_SAMPLES):
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return points


def normals_to_color(mesh):
    """Replace vertex colors with surface normals mapped to RGB."""
    normals = mesh.vertex_normals
    colors  = ((normals + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    alpha   = np.full((len(colors), 1), 255, dtype=np.uint8)
    colors  = np.concatenate([colors, alpha], axis=1)
    normal_mesh = mesh.copy()
    normal_mesh.visual = trimesh.visual.ColorVisuals(mesh=normal_mesh, vertex_colors=colors)
    return normal_mesh


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_mesh_views(mesh, camera_poses, normal_mode=False):
    """
    Render mesh from all camera poses.
    normal_mode=False → vertex color render  (PSNR / LPIPS)
    normal_mode=True  → surface normal render (PSNR-N / LPIPS-N)
    """
    render_mesh = normals_to_color(mesh) if normal_mode else mesh
    py_mesh = pyrender.Mesh.from_trimesh(render_mesh, smooth=False)
    r = pyrender.OffscreenRenderer(RESOLUTION, RESOLUTION)
    renders = []

    for cam_pose in camera_poses:
        scene = pyrender.Scene(
            ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
            bg_color=np.array([0.5, 0.5, 0.5, 1.0]),
        )
        scene.add(py_mesh)
        scene.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0), pose=cam_pose)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)
        renders.append(color)

    r.delete()
    return renders


# ── Image metrics ─────────────────────────────────────────────────────────────

def compute_psnr(img_a, img_b):
    a = img_a.astype(np.float32) / 255.0
    b = img_b.astype(np.float32) / 255.0
    return float(ski_psnr(a, b, data_range=1.0))


def to_lpips_tensor(img):
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(DEVICE) * 2.0 - 1.0


def compute_lpips_batch(renders_gt, renders_pred, loss_fn):
    scores = []
    for gt, pred in zip(renders_gt, renders_pred):
        with torch.no_grad():
            scores.append(float(loss_fn(to_lpips_tensor(gt), to_lpips_tensor(pred)).item()))
    return scores


def score_render_pair(gt_renders, pred_renders, loss_fn):
    psnr_scores  = [compute_psnr(gt_renders[j], pred_renders[j]) for j in range(len(gt_renders))]
    lpips_scores = compute_lpips_batch(gt_renders, pred_renders, loss_fn)
    return float(np.mean(psnr_scores)), float(np.mean(lpips_scores)), psnr_scores, lpips_scores


# ── 3D metrics ────────────────────────────────────────────────────────────────

def chamfer_distance(pts_a, pts_b):
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_a, _ = tree_b.query(pts_a)
    d_b, _ = tree_a.query(pts_b)
    return float(np.mean(d_a ** 2) + np.mean(d_b ** 2))


def f_score(pts_a, pts_b, threshold=THRESHOLD):
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_a, _ = tree_b.query(pts_a)
    d_b, _ = tree_a.query(pts_b)
    precision = np.mean(d_a < threshold)
    recall    = np.mean(d_b < threshold)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print("Loading LPIPS (AlexNet)...")
    loss_fn = lpips.LPIPS(net="alex").to(DEVICE)
    print("LPIPS loaded.\n")

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    with open(INF_MANIFEST) as f:
        inf_manifest = json.load(f)

    gt_paths = {}
    for subset in ["hard_set", "easy_set"]:
        for obj in split[subset]:
            gt_paths[obj["uid"]] = obj["glb_path"]

    results = {"hard": [], "easy": []}
    summary_accum = {
        "hard": {"cd": [], "fs": [], "psnr": [], "lpips": [], "psnr_n": [], "lpips_n": []},
        "easy": {"cd": [], "fs": [], "psnr": [], "lpips": [], "psnr_n": [], "lpips_n": []},
    }

    gt_renders_cache   = {}
    gt_renders_n_cache = {}

    for uid, info in inf_manifest.items():
        label      = info["label"]
        pred_paths = info["meshes"]
        gt_path    = gt_paths.get(uid)

        if not gt_path or not os.path.exists(gt_path):
            print(f"  SKIP {uid[:8]}: GT mesh not found")
            continue

        print(f"\n  Scoring {uid[:8]} ({label}, {len(pred_paths)} views)...")

        try:
            gt_mesh = load_and_normalize(gt_path)
            gt_pts  = sample_surface(gt_mesh)
        except Exception as e:
            print(f"  ERROR loading GT {uid[:8]}: {e}")
            continue

        try:
            if uid not in gt_renders_cache:
                gt_renders_cache[uid]   = render_mesh_views(gt_mesh, CAMERA_POSES, normal_mode=False)
            if uid not in gt_renders_n_cache:
                gt_renders_n_cache[uid] = render_mesh_views(gt_mesh, CAMERA_POSES, normal_mode=True)
        except Exception as e:
            print(f"  ERROR rendering GT {uid[:8]}: {e}")
            continue

        gt_renders   = gt_renders_cache[uid]
        gt_renders_n = gt_renders_n_cache[uid]
        per_view = []

        for i, pred_path in enumerate(pred_paths):
            if not os.path.exists(pred_path):
                print(f"    view{i:02d} SKIP: file not found")
                continue

            try:
                pred_mesh = load_and_normalize(pred_path)
                pred_pts  = sample_surface(pred_mesh)

                cd = chamfer_distance(pred_pts, gt_pts)
                fs = f_score(pred_pts, gt_pts)

                pred_renders   = render_mesh_views(pred_mesh, CAMERA_POSES, normal_mode=False)
                mean_psnr,  mean_lpips,  per_psnr,  per_lpips  = score_render_pair(gt_renders,   pred_renders,   loss_fn)

                pred_renders_n = render_mesh_views(pred_mesh, CAMERA_POSES, normal_mode=True)
                mean_psnr_n, mean_lpips_n, per_psnr_n, per_lpips_n = score_render_pair(gt_renders_n, pred_renders_n, loss_fn)

                per_view.append({
                    "input_view":         i,
                    "pred_path":          pred_path,
                    "chamfer_distance":   cd,
                    "f_score":            fs,
                    "mean_psnr":          mean_psnr,
                    "mean_lpips":         mean_lpips,
                    "mean_psnr_n":        mean_psnr_n,
                    "mean_lpips_n":       mean_lpips_n,
                    "per_render_psnr":    per_psnr,
                    "per_render_lpips":   per_lpips,
                    "per_render_psnr_n":  per_psnr_n,
                    "per_render_lpips_n": per_lpips_n,
                })

                print(f"    view{i:02d}  CD={cd:.4f}  F={fs:.4f}  "
                      f"PSNR={mean_psnr:.2f}  LPIPS={mean_lpips:.4f}  "
                      f"PSNR-N={mean_psnr_n:.2f}  LPIPS-N={mean_lpips_n:.4f}")

            except Exception as e:
                print(f"    ERROR view{i:02d}: {e}")

        if not per_view:
            continue

        obj_cd      = float(np.mean([v["chamfer_distance"] for v in per_view]))
        obj_fs      = float(np.mean([v["f_score"]          for v in per_view]))
        obj_psnr    = float(np.mean([v["mean_psnr"]        for v in per_view]))
        obj_lpips   = float(np.mean([v["mean_lpips"]       for v in per_view]))
        obj_psnr_n  = float(np.mean([v["mean_psnr_n"]      for v in per_view]))
        obj_lpips_n = float(np.mean([v["mean_lpips_n"]     for v in per_view]))

        entry = {
            "uid":          uid,
            "gt_path":      gt_path,
            "mean_chamfer": obj_cd,
            "mean_f_score": obj_fs,
            "mean_psnr":    obj_psnr,
            "mean_lpips":   obj_lpips,
            "mean_psnr_n":  obj_psnr_n,
            "mean_lpips_n": obj_lpips_n,
            "per_view":     per_view,
        }
        results[label].append(entry)
        a = summary_accum[label]
        a["cd"].append(obj_cd);         a["fs"].append(obj_fs)
        a["psnr"].append(obj_psnr);     a["lpips"].append(obj_lpips)
        a["psnr_n"].append(obj_psnr_n); a["lpips_n"].append(obj_lpips_n)

        print(f"  → avg  CD={obj_cd:.4f}  F={obj_fs:.4f}  "
              f"PSNR={obj_psnr:.2f}  LPIPS={obj_lpips:.4f}  "
              f"PSNR-N={obj_psnr_n:.2f}  LPIPS-N={obj_lpips_n:.4f}")

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else None

    summary = {}
    for label in ["hard", "easy"]:
        a = summary_accum[label]
        summary[label] = {
            "n":            len(a["cd"]),
            "mean_chamfer": safe_mean(a["cd"]),
            "mean_f_score": safe_mean(a["fs"]),
            "mean_psnr":    safe_mean(a["psnr"]),
            "mean_lpips":   safe_mean(a["lpips"]),
            "mean_psnr_n":  safe_mean(a["psnr_n"]),
            "mean_lpips_n": safe_mean(a["lpips_n"]),
        }

    output = {"summary": summary, "per_object": results}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    for label in ["hard", "easy"]:
        s = summary[label]
        print(f"{label.capitalize()} set (n={s['n']}):")
        print(f"  Chamfer Distance : {s['mean_chamfer']:.4f}")
        print(f"  F-score          : {s['mean_f_score']:.4f}")
        print(f"  PSNR             : {s['mean_psnr']:.2f} dB")
        print(f"  LPIPS            : {s['mean_lpips']:.4f}")
        print(f"  PSNR-N           : {s['mean_psnr_n']:.2f} dB")
        print(f"  LPIPS-N          : {s['mean_lpips_n']:.4f}")
    print(f"\nSaved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()