import json
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/TripoSR"))

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "inference_outputs")
MANIFEST_JSON = os.path.join(SCRIPT_DIR, "render_manifest.json")
SPLIT_JSON    = os.path.join(SCRIPT_DIR, "dataset_split.json")

os.makedirs(os.path.join(OUTPUT_DIR, "hard"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "easy"), exist_ok=True)

def rgba_to_rgb_white_bg(image):
    """Composite RGBA image onto white background to get RGB."""
    if image.mode != "RGBA":
        return image.convert("RGB")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    return bg.convert("RGB")

def main():
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    print("Loading TripoSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(131072)
    model.to("cuda")
    print("Model loaded.\n")

    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)
    with open(SPLIT_JSON) as f:
        split = json.load(f)

    results = {}

    for label in ["hard", "easy"]:
        print(f"Running inference on {label} set...")
        for uid, img_path in manifest[label].items():
            out_path = os.path.join(OUTPUT_DIR, label, f"{uid}.obj")

            if os.path.exists(out_path):
                print(f"  CACHED {uid[:8]}...")
                results[uid] = {"label": label, "mesh_path": out_path}
                continue

            print(f"  Processing {uid[:8]}...")
            try:
                # Load, remove background, composite onto white, convert to RGB
                image = Image.open(img_path).convert("RGBA")
                image = remove_background(image)
                image = resize_foreground(image, 0.85)
                image = rgba_to_rgb_white_bg(image)

                with torch.no_grad():
                    scene_codes = model([image], device="cuda")

                meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)
                meshes[0].export(out_path)

                results[uid] = {"label": label, "mesh_path": out_path}
                print(f"  Saved → {out_path}")

            except Exception as e:
                print(f"  ERROR {uid[:8]}: {e}")

    inf_manifest = os.path.expanduser("~/project_scripts/inference_manifest.json")
    with open(inf_manifest, "w") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    print(f"\nDone. Processed {total}/20 objects.")
    print(f"Meshes saved to {OUTPUT_DIR}")
    print(f"Inference manifest saved to {inf_manifest}")


if __name__ == "__main__":
    main()
