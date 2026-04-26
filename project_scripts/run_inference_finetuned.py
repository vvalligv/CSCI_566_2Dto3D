"""
run_inference_finetuned.py
Runs the depth-supervised finetuned TripoSR on each of the 8 input views per object.
Produces one OBJ mesh per view with vertex colors.

Reads:  input_view_manifest.json  (same input views as baseline)
Writes: inference_manifest_depth.json
"""
import json
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/TripoSR"))

from tsr.system import TSR

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "inference_outputs_depth")
INPUT_MANIFEST = os.path.join(SCRIPT_DIR, "input_view_manifest.json")  # same as baseline
SPLIT_JSON     = os.path.join(SCRIPT_DIR, "dataset_split.json")
CHECKPOINT     = os.path.join(SCRIPT_DIR, "checkpoints_depth", "final.ckpt")

os.makedirs(os.path.join(OUTPUT_DIR, "hard"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "easy"), exist_ok=True)


def preprocess_image(img_path):
    """Load pre-rendered PNG and composite onto gray background in float32."""
    image = Image.open(img_path).convert("RGBA")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def main():
    print("Loading fine-tuned TripoSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )

    # Load finetuned checkpoint on top of base weights
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint: {CHECKPOINT}")

    model.renderer.set_chunk_size(8192)
    model.to("cuda")
    print("Model loaded.\n")

    with open(INPUT_MANIFEST) as f:
        input_manifest = json.load(f)
    with open(SPLIT_JSON) as f:
        split = json.load(f)

    # Build uid -> label lookup
    uid_to_label = {}
    for subset, label in [("hard_set", "hard"), ("easy_set", "easy")]:
        for obj in split[subset]:
            uid_to_label[obj["uid"]] = label

    results = {}

    for label in ["hard", "easy"]:
        print(f"\nRunning finetuned inference on {label} set...")
        out_dir = os.path.join(OUTPUT_DIR, label)

        for uid, view_paths in input_manifest[label].items():
            obj_label = uid_to_label.get(uid, label)

            expected_meshes = [
                os.path.join(out_dir, f"{uid}_view{i:02d}.obj")
                for i in range(len(view_paths))
            ]
            # Change the cached block:
            if all(os.path.exists(p) for p in expected_meshes):
                print(f"  CACHED {uid[:8]}")
                results[uid] = {
                    "label": obj_label,
                    "meshes": expected_meshes,
                    "mesh_path": expected_meshes[0]   # add this
                }
                continue

            print(f"  Processing {uid[:8]} ({len(view_paths)} views)...")
            mesh_paths = []

            for i, img_path in enumerate(view_paths):
                out_mesh_path = os.path.join(out_dir, f"{uid}_view{i:02d}.obj")

                if os.path.exists(out_mesh_path):
                    print(f"    view{i:02d} CACHED")
                    mesh_paths.append(out_mesh_path)
                    continue

                try:
                    image = preprocess_image(img_path)

                    with torch.no_grad():
                        scene_codes = model([image], device="cuda")

                    meshes = model.extract_mesh(
                        scene_codes,
                        has_vertex_color=True,
                        resolution=256
                    )
                    meshes[0].export(out_mesh_path)
                    mesh_paths.append(out_mesh_path)
                    print(f"    view{i:02d} → {out_mesh_path}")

                except Exception as e:
                    print(f"    ERROR view{i:02d}: {e}")

            if mesh_paths:
                results[uid] = {
                    "label": obj_label,
                    "meshes": mesh_paths,
                    "mesh_path": mesh_paths[0]   # add this — metrics script uses this key
                }

    inf_manifest = os.path.join(SCRIPT_DIR, "inference_manifest_depth.json")
    with open(inf_manifest, "w") as f:
        json.dump(results, f, indent=2)

    total_meshes = sum(len(v["meshes"]) for v in results.values())
    print(f"\nDone. {len(results)}/20 objects, {total_meshes} total meshes.")
    print(f"Manifest → {inf_manifest}")


if __name__ == "__main__":
    main()