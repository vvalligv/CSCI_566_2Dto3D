"""
run_inference.py
Runs TripoSR on each of the 8 input views per object.
Produces one OBJ mesh per view with vertex colors (no separate texture file).

Output manifest: inference_manifest.json
  {
    "<uid>": {
      "label": "hard",
      "meshes": ["view00.obj", ..., "view07.obj"]
    }
  }
"""
import json
import os
import sys
import torch
import numpy as np
import rembg
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/TripoSR"))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "inference_outputs")
INPUT_MANIFEST = os.path.join(SCRIPT_DIR, "input_view_manifest.json")
SPLIT_JSON     = os.path.join(SCRIPT_DIR, "dataset_split.json")

os.makedirs(os.path.join(OUTPUT_DIR, "hard"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "easy"), exist_ok=True)


def preprocess_image(img_path):
    """
    Load pre-rendered PNG and composite onto gray background in float32.
    Matches TripoSR training expectations.
    """
    image = Image.open(img_path).convert("RGBA")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def main():
    print("Loading TripoSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
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
        print(f"\nRunning inference on {label} set...")
        out_dir = os.path.join(OUTPUT_DIR, label)

        for uid, view_paths in input_manifest[label].items():
            obj_label = uid_to_label.get(uid, label)

            # Check if all view meshes already exist
            expected_meshes = [
                os.path.join(out_dir, f"{uid}_view{i:02d}.obj")
                for i in range(len(view_paths))
            ]
            if all(os.path.exists(p) for p in expected_meshes):
                print(f"  CACHED {uid[:8]}")
                results[uid] = {"label": obj_label, "meshes": expected_meshes}
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

                    # has_vertex_color=True — colors baked directly into vertices
                    # no separate texture file needed
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
                results[uid] = {"label": obj_label, "meshes": mesh_paths}

    inf_manifest = os.path.join(SCRIPT_DIR, "inference_manifest.json")
    with open(inf_manifest, "w") as f:
        json.dump(results, f, indent=2)

    total_meshes = sum(len(v["meshes"]) for v in results.values())
    print(f"\nDone. {len(results)}/20 objects, {total_meshes} total meshes.")
    print(f"Manifest → {inf_manifest}")


if __name__ == "__main__":
    main()