import json
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SPLIT_JSON    = os.path.join(SCRIPT_DIR, "dataset_split.json")
INF_MANIFEST  = os.path.join(SCRIPT_DIR, "inference_manifest_depth.json")
BASELINE_JSON = os.path.join(SCRIPT_DIR, "metrics_baseline.json")
OUTPUT_JSON   = os.path.join(SCRIPT_DIR, "metrics_depth.json")
COMPARE_JSON  = os.path.join(SCRIPT_DIR, "etrics_comparison.json")
N_SAMPLES     = 10000

def sample_surface(mesh_path, n=N_SAMPLES):
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return points

def chamfer_distance(pts_a, pts_b):
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    return float(np.mean(dist_a**2) + np.mean(dist_b**2))

def f_score(pts_a, pts_b, threshold=0.05):
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    precision = np.mean(dist_a < threshold)
    recall    = np.mean(dist_b < threshold)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))

def main():
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    with open(INF_MANIFEST) as f:
        inf_manifest = json.load(f)
    with open(BASELINE_JSON) as f:
        baseline = json.load(f)

    baseline_lookup = {}
    for label in ["hard", "easy"]:
        for obj in baseline["per_object"][label]:
            baseline_lookup[obj["uid"]] = obj

    gt_paths = {}
    for subset in ["hard_set", "easy_set"]:
        for obj in split[subset]:
            gt_paths[obj["uid"]] = obj["glb_path"]

    results = {"hard": [], "easy": []}
    all_hard_cd, all_easy_cd = [], []
    all_hard_fs, all_easy_fs = [], []

    for uid, info in inf_manifest.items():
        label     = info["label"]
        pred_path = info["mesh_path"]
        gt_path   = gt_paths.get(uid)

        if gt_path is None or not os.path.exists(pred_path):
            continue

        print(f"  Scoring {uid[:8]}... ({label})")
        try:
            pred_pts = sample_surface(pred_path)
            gt_pts   = sample_surface(gt_path)
            cd       = chamfer_distance(pred_pts, gt_pts)
            fs       = f_score(pred_pts, gt_pts, threshold=0.05)

            base     = baseline_lookup.get(uid, {})
            base_cd  = base.get("chamfer_distance", None)
            base_fs  = base.get("f_score", None)

            cd_delta = (cd - base_cd) if base_cd else None
            fs_delta = (fs - base_fs) if base_fs else None

            entry = {
                "uid": uid,
                "chamfer_distance": cd,
                "f_score": fs,
                "baseline_cd": base_cd,
                "baseline_fs": base_fs,
                "cd_delta": cd_delta,
                "fs_delta": fs_delta,
            }
            results[label].append(entry)

            if label == "hard":
                all_hard_cd.append(cd)
                all_hard_fs.append(fs)
            else:
                all_easy_cd.append(cd)
                all_easy_fs.append(fs)

            delta_str = f"  ΔCD={cd_delta:+.4f}  ΔF={fs_delta:+.4f}" if cd_delta is not None else ""
            print(f"    CD={cd:.4f}  F={fs:.4f}{delta_str}")

        except Exception as e:
            print(f"  ERROR {uid[:8]}: {e}")

    summary = {
        "hard": {
            "mean_chamfer": float(np.mean(all_hard_cd)) if all_hard_cd else None,
            "mean_f_score": float(np.mean(all_hard_fs)) if all_hard_fs else None,
            "n": len(all_hard_cd)
        },
        "easy": {
            "mean_chamfer": float(np.mean(all_easy_cd)) if all_easy_cd else None,
            "mean_f_score": float(np.mean(all_easy_fs)) if all_easy_fs else None,
            "n": len(all_easy_cd)
        }
    }

    base_summary = baseline["summary"]

    output = {"summary": summary, "per_object": results}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPARISON: Baseline vs Depth Supervision Fine-tuned")
    print(f"{'='*60}")
    print(f"{'':20} {'Baseline':>15} {'Depth FT':>15} {'Delta':>10}")
    print(f"{'-'*60}")
    print(f"{'Hard CD':20} {base_summary['hard']['mean_chamfer']:>15.4f} "
          f"{summary['hard']['mean_chamfer']:>15.4f} "
          f"{summary['hard']['mean_chamfer'] - base_summary['hard']['mean_chamfer']:>+10.4f}")
    print(f"{'Hard F-score':20} {base_summary['hard']['mean_f_score']:>15.4f} "
          f"{summary['hard']['mean_f_score']:>15.4f} "
          f"{summary['hard']['mean_f_score'] - base_summary['hard']['mean_f_score']:>+10.4f}")
    print(f"{'-'*60}")
    print(f"{'Easy CD':20} {base_summary['easy']['mean_chamfer']:>15.4f} "
          f"{summary['easy']['mean_chamfer']:>15.4f} "
          f"{summary['easy']['mean_chamfer'] - base_summary['easy']['mean_chamfer']:>+10.4f}")
    print(f"{'Easy F-score':20} {base_summary['easy']['mean_f_score']:>15.4f} "
          f"{summary['easy']['mean_f_score']:>15.4f} "
          f"{summary['easy']['mean_f_score'] - base_summary['easy']['mean_f_score']:>+10.4f}")
    print(f"{'='*60}")
    print(f"\nSaved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
