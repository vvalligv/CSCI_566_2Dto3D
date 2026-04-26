"""
compute_metrics_finetuned.py
Scores each predicted mesh from the depth-supervised finetuned model
against the GT GLB. Computes deltas against baseline.

Metrics per object:
  3D:    Chamfer Distance, F-score
  Image: PSNR, LPIPS       (color/vertex renders)
         PSNR-N, LPIPS-N   (surface normal renders)

Output: metrics_depth.json
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

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SPLIT_JSON    = os.path.join(SCRIPT_DIR, "dataset_split.json")
INF_MANIFEST  = os.path.join(SCRIPT_DIR, "inference_manifest_depth.json")
BASELINE_JSON = os.path.join(SCRIPT_DIR, "metrics_baseline.json")
OUTPUT_JSON   = os.path.join(SCRIPT_DIR, "metrics_depth.json")

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
        azimuth  = 2 * np.pi * i / n_views
        x = cam_dist * np.cos(elevation) * np.sin(azimuth)
        y = cam_dist * np.sin(elevation)
        z = cam_dist * np.cos(elevation) * np.cos(azimuth)
        cam_pos = np.array([x, y, z])

        forward  = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 1.0, 0.0])
        right    = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 0.0, 1.0])
            right    = np.cross(forward, world_up)
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
    normals = mesh.vertex_normals
    colors  = ((normals + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    alpha   = np.full((len(colors), 1), 255, dtype=np.uint8)
    colors  = np.concatenate([colors, alpha], axis=1)
    normal_mesh = mesh.copy()
    normal_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=normal_mesh, vertex_colors=colors
    )
    return normal_mesh


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_mesh_views(mesh, camera_poses, normal_mode=False):
    render_mesh = normals_to_color(mesh) if normal_mode else mesh
    py_mesh     = pyrender.Mesh.from_trimesh(render_mesh, smooth=False)
    r           = pyrender.OffscreenRenderer(RESOLUTION, RESOLUTION)
    renders     = []

    for cam_pose in camera_poses:
        scene = pyrender.Scene(
            ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
            bg_color=np.array([0.5, 0.5, 0.5, 1.0]),
        )
        scene.add(py_mesh)
        scene.add(
            pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0),
            pose=cam_pose,
        )
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
            scores.append(
                float(loss_fn(to_lpips_tensor(gt), to_lpips_tensor(pred)).item())
            )
    return scores


def score_render_pair(gt_renders, pred_renders, loss_fn):
    psnr_scores  = [
        compute_psnr(gt_renders[j], pred_renders[j])
        for j in range(len(gt_renders))
    ]
    lpips_scores = compute_lpips_batch(gt_renders, pred_renders, loss_fn)
    return (
        float(np.mean(psnr_scores)),
        float(np.mean(lpips_scores)),
        psnr_scores,
        lpips_scores,
    )


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
    with open(BASELINE_JSON) as f:
        baseline = json.load(f)

    # Build baseline lookup keyed by uid
    # Baseline per-object keys: mean_chamfer, mean_f_score, mean_psnr,
    #                            mean_lpips, mean_psnr_n, mean_lpips_n
    baseline_lookup = {}
    for label in ["hard", "easy"]:
        for obj in baseline["per_object"][label]:
            baseline_lookup[obj["uid"]] = obj

    gt_paths = {}
    for subset in ["hard_set", "easy_set"]:
        for obj in split[subset]:
            gt_paths[obj["uid"]] = obj["glb_path"]

    results       = {"hard": [], "easy": []}
    summary_accum = {
        "hard": {"cd": [], "fs": [], "psnr": [], "lpips": [], "psnr_n": [], "lpips_n": []},
        "easy": {"cd": [], "fs": [], "psnr": [], "lpips": [], "psnr_n": [], "lpips_n": []},
    }

    gt_renders_cache   = {}
    gt_renders_n_cache = {}

    for uid, info in inf_manifest.items():
        label     = info["label"]
        mesh_path = info["mesh_path"]
        gt_path   = gt_paths.get(uid)

        if not gt_path or not os.path.exists(gt_path):
            print(f"  SKIP {uid[:8]}: GT mesh not found")
            continue
        if not os.path.exists(mesh_path):
            print(f"  SKIP {uid[:8]}: predicted mesh not found at {mesh_path}")
            continue

        print(f"\n  Scoring {uid[:8]} ({label})...")

        # ── Load meshes ───────────────────────────────────────────────────────
        try:
            gt_mesh   = load_and_normalize(gt_path)
            gt_pts    = sample_surface(gt_mesh)
            pred_mesh = load_and_normalize(mesh_path)
            pred_pts  = sample_surface(pred_mesh)
        except Exception as e:
            print(f"  ERROR loading meshes {uid[:8]}: {e}")
            continue

        # ── 3D metrics ────────────────────────────────────────────────────────
        cd = chamfer_distance(pred_pts, gt_pts)
        fs = f_score(pred_pts, gt_pts)

        # ── Render GT views (cached across objects) ───────────────────────────
        try:
            if uid not in gt_renders_cache:
                gt_renders_cache[uid] = render_mesh_views(
                    gt_mesh, CAMERA_POSES, normal_mode=False
                )
            if uid not in gt_renders_n_cache:
                gt_renders_n_cache[uid] = render_mesh_views(
                    gt_mesh, CAMERA_POSES, normal_mode=True
                )
        except Exception as e:
            print(f"  ERROR rendering GT {uid[:8]}: {e}")
            continue

        gt_renders   = gt_renders_cache[uid]
        gt_renders_n = gt_renders_n_cache[uid]

        # ── Render predicted views ────────────────────────────────────────────
        try:
            pred_renders   = render_mesh_views(pred_mesh, CAMERA_POSES, normal_mode=False)
            pred_renders_n = render_mesh_views(pred_mesh, CAMERA_POSES, normal_mode=True)

            mean_psnr,   mean_lpips,   per_psnr,   per_lpips   = score_render_pair(
                gt_renders,   pred_renders,   loss_fn
            )
            mean_psnr_n, mean_lpips_n, per_psnr_n, per_lpips_n = score_render_pair(
                gt_renders_n, pred_renders_n, loss_fn
            )
        except Exception as e:
            print(f"  ERROR rendering pred {uid[:8]}: {e}")
            continue

        # ── Baseline deltas (baseline uses mean_chamfer / mean_f_score etc.) ──
        base        = baseline_lookup.get(uid, {})
        base_cd     = base.get("mean_chamfer")
        base_fs     = base.get("mean_f_score")
        base_psnr   = base.get("mean_psnr")
        base_lpips  = base.get("mean_lpips")
        base_psnr_n  = base.get("mean_psnr_n")
        base_lpips_n = base.get("mean_lpips_n")

        def delta(new, old):
            return (new - old) if (new is not None and old is not None) else None

        entry = {
            "uid":          uid,
            "gt_path":      gt_path,
            "mesh_path":    mesh_path,
            # 3D
            "mean_chamfer":     cd,
            "mean_f_score":     fs,
            "baseline_cd":      base_cd,
            "baseline_fs":      base_fs,
            "cd_delta":         delta(cd, base_cd),
            "fs_delta":         delta(fs, base_fs),
            # Image
            "mean_psnr":        mean_psnr,
            "mean_lpips":       mean_lpips,
            "mean_psnr_n":      mean_psnr_n,
            "mean_lpips_n":     mean_lpips_n,
            "baseline_psnr":    base_psnr,
            "baseline_lpips":   base_lpips,
            "baseline_psnr_n":  base_psnr_n,
            "baseline_lpips_n": base_lpips_n,
            "psnr_delta":       delta(mean_psnr,   base_psnr),
            "lpips_delta":      delta(mean_lpips,  base_lpips),
            "psnr_n_delta":     delta(mean_psnr_n, base_psnr_n),
            "lpips_n_delta":    delta(mean_lpips_n,base_lpips_n),
            "per_render_psnr":    per_psnr,
            "per_render_lpips":   per_lpips,
            "per_render_psnr_n":  per_psnr_n,
            "per_render_lpips_n": per_lpips_n,
        }

        results[label].append(entry)
        a = summary_accum[label]
        a["cd"].append(cd);              a["fs"].append(fs)
        a["psnr"].append(mean_psnr);     a["lpips"].append(mean_lpips)
        a["psnr_n"].append(mean_psnr_n); a["lpips_n"].append(mean_lpips_n)

        def fmt_delta(v, fmt=".4f"):
            return f"{v:+{fmt}}" if v is not None else "N/A"

        print(f"  CD={cd:.4f} (Δ{fmt_delta(entry['cd_delta'])})  "
              f"F={fs:.4f} (Δ{fmt_delta(entry['fs_delta'])})  "
              f"PSNR={mean_psnr:.2f} (Δ{fmt_delta(entry['psnr_delta'], '.2f')})  "
              f"LPIPS={mean_lpips:.4f} (Δ{fmt_delta(entry['lpips_delta'])})  "
              f"PSNR-N={mean_psnr_n:.2f} (Δ{fmt_delta(entry['psnr_n_delta'], '.2f')})  "
              f"LPIPS-N={mean_lpips_n:.4f} (Δ{fmt_delta(entry['lpips_n_delta'])})")

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

    # ── Comparison table ──────────────────────────────────────────────────────
    base_s = baseline["summary"]
    print(f"\n{'='*70}")
    print(f"COMPARISON: Baseline vs Depth Supervision Fine-tuned")
    print(f"{'='*70}")
    print(f"{'':22} {'Baseline':>12} {'Depth FT':>12} {'Delta':>10}  {'Better?':>7}")
    print(f"{'-'*70}")

    for label in ["hard", "easy"]:
        s = summary[label]
        b = base_s[label]
        print(f"\n  {label.upper()} SET (n={s['n']})")
        for metric, key, fmt, higher_better in [
            ("Chamfer Dist",  "mean_chamfer", ".4f", False),
            ("F-score",       "mean_f_score", ".4f", True),
            ("PSNR",          "mean_psnr",    ".2f", True),
            ("LPIPS",         "mean_lpips",   ".4f", False),
            ("PSNR-N",        "mean_psnr_n",  ".2f", True),
            ("LPIPS-N",       "mean_lpips_n", ".4f", False),
        ]:
            bv = b.get(key)
            sv = s.get(key)
            if bv is None or sv is None:
                continue
            d      = sv - bv
            better = (d > 0) if higher_better else (d < 0)
            tag    = "✓" if better else "✗"
            print(f"  {metric:20} {bv:>12{fmt}} {sv:>12{fmt}} {d:>+10{fmt}}  {tag}")

    print(f"\n{'='*70}")
    print(f"\nSaved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()