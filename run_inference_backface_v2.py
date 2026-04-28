"""
run_inference_backface_v2.py — Run inference with the fine-tuned checkpoint.

Loads the checkpoint, runs on all objects in the dataset split, and exports
predicted meshes as .obj files. Uses adaptive thresholding for robust mesh
extraction across varying density scales.

Usage:
  python run_inference_backface_v2.py --base-dir /path/to/project
"""

import argparse
import json
import os
import sys
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Inference with back-face fine-tuned TripoSR")
    p.add_argument("--base-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                    help="Project root (contains TripoSR/, checkpoints, etc.)")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint (default: <base-dir>/checkpoints_backface_v2/final.ckpt)")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Where to save meshes (default: <base-dir>/inference_outputs_backface_v2)")
    p.add_argument("--split-json", type=str, default=None)
    p.add_argument("--render-manifest", type=str, default=None)
    p.add_argument("--resolution", type=int, default=256, help="Marching cubes resolution")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def rgba_to_rgb_white_bg(image):
    if image.mode != "RGBA":
        return image.convert("RGB")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    return bg.convert("RGB")


def main():
    args = parse_args()
    base = args.base_dir
    data_dir = os.path.join(base, "project_scripts/project_scripts")

    checkpoint = args.checkpoint or os.path.join(base, "checkpoints_backface_v2/final.ckpt")
    output_dir = args.output_dir or os.path.join(base, "inference_outputs_backface_v2")
    split_json = args.split_json or os.path.join(data_dir, "dataset_split.json")
    manifest_json = args.render_manifest or os.path.join(data_dir, "render_manifest.json")
    inf_manifest = os.path.join(data_dir, "inference_manifest_backface_v2.json")

    sys.path.insert(0, os.path.join(base, "TripoSR"))
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    os.makedirs(os.path.join(output_dir, "hard"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "easy"), exist_ok=True)

    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found at {checkpoint}")
        print("Run finetune_backface_v2.py first.")
        return

    print("Loading fine-tuned TripoSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt)
    model.renderer.set_chunk_size(131072)
    model.to(args.device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}\n")

    with open(manifest_json) as f:
        manifest = json.load(f)

    results = {}

    for label in ["hard", "easy"]:
        print(f"Running inference on {label} set...")
        for uid, img_path in manifest[label].items():
            out_path = os.path.join(output_dir, label, f"{uid}.obj")

            if os.path.exists(out_path):
                print(f"  CACHED {uid[:8]}...")
                results[uid] = {"label": label, "mesh_path": out_path}
                continue

            print(f"  Processing {uid[:8]}...")
            try:
                image = Image.open(img_path).convert("RGBA")
                image = remove_background(image)
                image = resize_foreground(image, 0.85)
                image = rgba_to_rgb_white_bg(image)

                with torch.no_grad():
                    scene_codes = model([image], device=args.device)

                for thresh in [10.0, 5.0, 2.0, 1.0]:
                    try:
                        meshes = model.extract_mesh(
                            scene_codes, has_vertex_color=True,
                            resolution=args.resolution, threshold=thresh)
                        if meshes[0].vertices.shape[0] > 0:
                            meshes[0].export(out_path)
                            results[uid] = {"label": label, "mesh_path": out_path}
                            print(f"  Saved -> {out_path} (thresh={thresh})")
                            break
                    except Exception:
                        continue
                else:
                    print(f"  FAIL {uid[:8]}: no surface at any threshold")

            except Exception as e:
                print(f"  ERROR {uid[:8]}: {e}")

    with open(inf_manifest, "w") as f:
        json.dump(results, f, indent=2)

    total = len(manifest.get("hard", {})) + len(manifest.get("easy", {}))
    print(f"\nDone. Processed {len(results)}/{total} objects.")
    print(f"Meshes saved to {output_dir}")
    print(f"Manifest saved to {inf_manifest}")


if __name__ == "__main__":
    main()
