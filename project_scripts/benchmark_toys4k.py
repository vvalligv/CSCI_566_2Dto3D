import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.expanduser("~/TripoSR"))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def rgba_to_rgb_white_bg(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    return bg.convert("RGB")


def sample_surface(mesh_path: str, n_samples: int) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))
    points, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return points


def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    return float(np.mean(dist_a ** 2) + np.mean(dist_b ** 2))


def f_score(pts_a: np.ndarray, pts_b: np.ndarray, threshold: float) -> float:
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    precision = np.mean(dist_a < threshold)
    recall = np.mean(dist_b < threshold)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def parse_split_file(split_file: Path):
    with split_file.open() as f:
        data = json.load(f)

    id_to_split = {}

    if isinstance(data, dict):
        for split_name, ids in data.items():
            if isinstance(ids, list):
                for sample_id in ids:
                    id_to_split[str(sample_id)] = str(split_name)
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "id" in entry:
                id_to_split[str(entry["id"])] = str(entry.get("split", "test"))

    return id_to_split


def resolve_image_path(dataset_root: Path, sample_id: str, image_templates):
    for template in image_templates:
        rel = template.format(id=sample_id)
        if "*" in rel or "?" in rel:
            matches = sorted(dataset_root.glob(rel))
            if matches:
                return matches[0]
        else:
            candidate = dataset_root / rel
            if candidate.exists():
                return candidate
    return None


def discover_samples(args):
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    mesh_paths = sorted(dataset_root.glob(args.mesh_glob))
    split_lookup = {}
    if args.split_file:
        split_lookup = parse_split_file(Path(args.split_file).expanduser().resolve())

    samples = []
    for mesh_path in mesh_paths:
        sample_id = mesh_path.stem
        image_path = resolve_image_path(dataset_root, sample_id, args.image_template)
        if image_path is None:
            continue
        split = split_lookup.get(sample_id, args.default_split)
        samples.append(
            {
                "id": sample_id,
                "split": split,
                "image_path": str(image_path.resolve()),
                "gt_mesh_path": str(mesh_path.resolve()),
            }
        )

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    return samples


def run_inference(samples, args, pred_dir: Path):
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    pred_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading TripoSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).expanduser().resolve()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {ckpt_path}")

    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    model.eval()

    manifest = {}

    for i, sample in enumerate(samples, start=1):
        sample_id = sample["id"]
        out_path = pred_dir / f"{sample_id}.obj"

        print(f"[{i}/{len(samples)}] {sample_id} ({sample['split']})")

        if out_path.exists() and not args.force:
            manifest[sample_id] = {
                **sample,
                "pred_mesh_path": str(out_path),
            }
            print("  CACHED")
            continue

        try:
            image = Image.open(sample["image_path"]).convert("RGBA")
            if args.remove_background:
                image = remove_background(image)
                image = resize_foreground(image, args.foreground_ratio)
            image = rgba_to_rgb_white_bg(image)

            with torch.no_grad():
                scene_codes = model([image], device=device)

            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=args.mesh_resolution,
            )
            meshes[0].export(str(out_path))

            manifest[sample_id] = {
                **sample,
                "pred_mesh_path": str(out_path),
            }
            print(f"  Saved -> {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    return manifest


def compute_metrics(manifest, args):
    results = []
    per_split = {}

    for i, (sample_id, info) in enumerate(manifest.items(), start=1):
        print(f"Scoring [{i}/{len(manifest)}] {sample_id}...")
        try:
            pred_pts = sample_surface(info["pred_mesh_path"], args.n_samples)
            gt_pts = sample_surface(info["gt_mesh_path"], args.n_samples)
            cd = chamfer_distance(pred_pts, gt_pts)
            fs = f_score(pred_pts, gt_pts, threshold=args.fscore_threshold)

            row = {
                "id": sample_id,
                "split": info["split"],
                "image_path": info["image_path"],
                "gt_mesh_path": info["gt_mesh_path"],
                "pred_mesh_path": info["pred_mesh_path"],
                "chamfer_distance": cd,
                "f_score": fs,
            }
            results.append(row)
            per_split.setdefault(info["split"], {"cd": [], "fs": []})
            per_split[info["split"]]["cd"].append(cd)
            per_split[info["split"]]["fs"].append(fs)

            print(f"  CD={cd:.6f} F={fs:.6f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    summary = {
        "overall": {
            "n": len(results),
            "mean_chamfer": float(np.mean([r["chamfer_distance"] for r in results])) if results else None,
            "mean_f_score": float(np.mean([r["f_score"] for r in results])) if results else None,
        },
        "by_split": {},
    }

    for split_name, split_vals in per_split.items():
        summary["by_split"][split_name] = {
            "n": len(split_vals["cd"]),
            "mean_chamfer": float(np.mean(split_vals["cd"])) if split_vals["cd"] else None,
            "mean_f_score": float(np.mean(split_vals["fs"])) if split_vals["fs"] else None,
        }

    return {"summary": summary, "per_object": results}


def build_parser():
    parser = argparse.ArgumentParser(description="Run a Toys4k benchmark with TripoSR.")
    parser.add_argument("--dataset-root", required=True, help="Root folder of the Toys4k dataset.")
    parser.add_argument(
        "--mesh-glob",
        default="quant/**/*.obj",
        help="Glob (relative to dataset root) for GT meshes.",
    )
    parser.add_argument(
        "--image-template",
        action="append",
        default=["renders_cond/{id}/000.png", "renders_cond/{id}/*.png", "renders_cond/{id}/*.jpg"],
        help="Relative image path template(s). Supports {id}; wildcards are allowed.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional JSON split file ({split:[ids]} or [{id,split}]).",
    )
    parser.add_argument("--default-split", default="test", help="Split label when split file is missing.")
    parser.add_argument("--output-dir", default="./toys4k_benchmark", help="Output directory.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples for smoke tests.")
    parser.add_argument("--checkpoint", default=None, help="Optional fine-tuned checkpoint path.")
    parser.add_argument("--mesh-resolution", type=int, default=256, help="Marching cubes resolution.")
    parser.add_argument("--chunk-size", type=int, default=131072, help="Renderer chunk size.")
    parser.add_argument("--foreground-ratio", type=float, default=0.85, help="Foreground resize ratio.")
    parser.add_argument(
        "--remove-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TripoSR background removal preprocessing.",
    )
    parser.add_argument("--n-samples", type=int, default=10000, help="Surface points sampled per mesh.")
    parser.add_argument("--fscore-threshold", type=float, default=0.05, help="F-score distance threshold.")
    parser.add_argument("--force", action="store_true", help="Re-run inference even if output mesh exists.")
    parser.add_argument(
        "--variant",
        default=None,
        help="Run label stored in metrics_toys4k.json (defaults to pretrained or checkpoint stem).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.variant:
        variant = args.variant
    elif args.checkpoint:
        variant = Path(args.checkpoint).stem
    else:
        variant = "pretrained"

    run_dir = output_dir / "runs" / variant
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(args)
    if not samples:
        print("No valid Toys4k samples found. Check --mesh-glob and --image-template.")
        return

    print(f"Discovered {len(samples)} samples.")

    sample_manifest_path = run_dir / "samples.json"
    with sample_manifest_path.open("w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved sample list -> {sample_manifest_path}")

    pred_dir = run_dir / "pred_meshes"
    manifest = run_inference(samples, args, pred_dir)
    inf_manifest_path = run_dir / "inference_manifest.json"
    with inf_manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved inference manifest -> {inf_manifest_path}")

    metrics = compute_metrics(manifest, args)
    run_metrics_path = run_dir / "metrics.json"
    with run_metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    metrics_path = output_dir / "metrics_toys4k.json"
    if metrics_path.exists():
        with metrics_path.open() as f:
            metrics_store = json.load(f)
        if not isinstance(metrics_store, dict):
            metrics_store = {}
    else:
        metrics_store = {}

    runs = metrics_store.get("runs", {})
    runs[variant] = {
        "config": {
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None,
            "mesh_glob": args.mesh_glob,
            "image_template": args.image_template,
            "n_samples": args.n_samples,
            "fscore_threshold": args.fscore_threshold,
            "mesh_resolution": args.mesh_resolution,
            "chunk_size": args.chunk_size,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "samples": str(sample_manifest_path),
            "inference_manifest": str(inf_manifest_path),
            "pred_meshes_dir": str(pred_dir),
            "metrics": str(run_metrics_path),
        },
        **metrics,
    }
    metrics_store["runs"] = runs

    with metrics_path.open("w") as f:
        json.dump(metrics_store, f, indent=2)

    print("\n" + "=" * 60)
    print("TOYS4K BENCHMARK RESULTS")
    print("=" * 60)
    overall = metrics["summary"]["overall"]
    print(f"Samples: {overall['n']}")
    print(f"Mean Chamfer Distance: {overall['mean_chamfer']}")
    print(f"Mean F-score: {overall['mean_f_score']}")
    print(f"Predicted meshes dir: {pred_dir}")
    if metrics["summary"]["by_split"]:
        print("\nBy split:")
        for split_name, split_vals in metrics["summary"]["by_split"].items():
            print(
                f"  {split_name}: n={split_vals['n']}, "
                f"CD={split_vals['mean_chamfer']}, F={split_vals['mean_f_score']}"
            )
    print(f"\nSaved metrics ({variant}) -> {metrics_path}")


if __name__ == "__main__":
    main()
