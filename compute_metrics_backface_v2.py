"""
compute_metrics_backface_v2.py — Compute Chamfer Distance and F-score for the
fine-tuned model and print a comparison table against baseline (and v1 if available).

Usage:
  python compute_metrics_backface_v2.py --base-dir /path/to/project
"""

import argparse
import json
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser(description="Compute CD and F-score for back-face v2")
    p.add_argument("--base-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                    help="Project root")
    p.add_argument("--split-json", type=str, default=None)
    p.add_argument("--inf-manifest", type=str, default=None,
                    help="Inference manifest JSON from run_inference_backface_v2.py")
    p.add_argument("--baseline-json", type=str, default=None,
                    help="Baseline metrics JSON for comparison")
    p.add_argument("--v1-json", type=str, default=None,
                    help="v1 metrics JSON for comparison (optional)")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--n-samples", type=int, default=10000,
                    help="Surface sample count for metric computation")
    p.add_argument("--f-threshold", type=float, default=0.05,
                    help="Distance threshold for F-score")
    return p.parse_args()


def sample_surface(mesh_path, n):
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return points


def chamfer_distance(pts_a, pts_b):
    tree_a, tree_b = cKDTree(pts_a), cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    return float(np.mean(dist_a**2) + np.mean(dist_b**2))


def f_score(pts_a, pts_b, threshold):
    tree_a, tree_b = cKDTree(pts_a), cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    precision = np.mean(dist_a < threshold)
    recall = np.mean(dist_b < threshold)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def main():
    args = parse_args()
    base = args.base_dir
    data_dir = os.path.join(base, "project_scripts/project_scripts")

    split_json = args.split_json or os.path.join(data_dir, "dataset_split.json")
    inf_manifest = args.inf_manifest or os.path.join(data_dir, "inference_manifest_backface_v2.json")
    baseline_json = args.baseline_json or os.path.join(data_dir, "metrics_baseline.json")
    v1_json = args.v1_json or os.path.join(data_dir, "metrics_backface.json")
    output_json = args.output_json or os.path.join(data_dir, "metrics_backface_v2.json")

    with open(split_json) as f:
        split = json.load(f)
    with open(inf_manifest) as f:
        inf_data = json.load(f)
    with open(baseline_json) as f:
        baseline = json.load(f)

    v1_data = None
    if os.path.exists(v1_json):
        with open(v1_json) as f:
            v1_data = json.load(f)

    baseline_lookup = {}
    for label in ["hard", "easy"]:
        for obj in baseline["per_object"][label]:
            baseline_lookup[obj["uid"]] = obj

    v1_lookup = {}
    if v1_data:
        for label in ["hard", "easy"]:
            for obj in v1_data["per_object"][label]:
                v1_lookup[obj["uid"]] = obj

    gt_paths = {}
    for subset in ["hard_set", "easy_set"]:
        for obj in split[subset]:
            gt_paths[obj["uid"]] = obj["glb_path"]

    results = {"hard": [], "easy": []}
    scores = {k: {"cd": [], "fs": []} for k in ["hard", "easy"]}

    for uid, info in inf_data.items():
        label = info["label"]
        pred_path = info["mesh_path"]
        gt_path = gt_paths.get(uid)

        if gt_path is None or not os.path.exists(pred_path):
            print(f"  SKIP {uid[:8]}: missing files")
            continue

        print(f"  Scoring {uid[:8]}... ({label})")
        try:
            pred_pts = sample_surface(pred_path, args.n_samples)
            gt_pts = sample_surface(gt_path, args.n_samples)
            cd = chamfer_distance(pred_pts, gt_pts)
            fs = f_score(pred_pts, gt_pts, args.f_threshold)

            bl = baseline_lookup.get(uid, {})
            v1 = v1_lookup.get(uid, {})

            results[label].append({
                "uid": uid, "chamfer_distance": cd, "f_score": fs,
                "baseline_cd": bl.get("chamfer_distance"),
                "baseline_fs": bl.get("f_score"),
                "v1_cd": v1.get("chamfer_distance"),
                "v1_fs": v1.get("f_score"),
            })
            scores[label]["cd"].append(cd)
            scores[label]["fs"].append(fs)
            print(f"    CD={cd:.4f}  F={fs:.4f}")

        except Exception as e:
            print(f"  ERROR {uid[:8]}: {e}")

    summary = {}
    for label in ["hard", "easy"]:
        summary[label] = {
            "mean_chamfer": float(np.mean(scores[label]["cd"])) if scores[label]["cd"] else None,
            "mean_f_score": float(np.mean(scores[label]["fs"])) if scores[label]["fs"] else None,
            "n": len(scores[label]["cd"]),
        }

    with open(output_json, "w") as f:
        json.dump({"summary": summary, "per_object": results}, f, indent=2)

    # ── Comparison table ──────────────────────────────────────────────────
    bs = baseline["summary"]
    vs = v1_data["summary"] if v1_data else {}

    def fmt(v):
        return f"{v:>12.4f}" if v is not None else f"{'N/A':>12}"
    def delta(new, old):
        return f"{new - old:>+12.4f}" if (new is not None and old is not None) else f"{'N/A':>12}"

    print(f"\n{'='*80}")
    print("COMPARISON: Baseline vs Back-Face v1 vs Back-Face v2")
    print(f"{'='*80}")
    print(f"{'':20} {'Baseline':>12} {'BF v1':>12} {'BF v2':>12} {'d(v2-base)':>12}")
    print(f"{'-'*80}")
    for label in ["hard", "easy"]:
        bc = bs[label]["mean_chamfer"]
        bf = bs[label]["mean_f_score"]
        vc = vs.get(label, {}).get("mean_chamfer")
        vf = vs.get(label, {}).get("mean_f_score")
        c2 = summary[label]["mean_chamfer"]
        f2 = summary[label]["mean_f_score"]
        tag = label.capitalize()
        print(f"{tag + ' CD':20}{fmt(bc)}{fmt(vc)}{fmt(c2)}{delta(c2, bc)}")
        print(f"{tag + ' F-score':20}{fmt(bf)}{fmt(vf)}{fmt(f2)}{delta(f2, bf)}")
        if label == "hard":
            print(f"{'-'*80}")
    print(f"{'='*80}")
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
