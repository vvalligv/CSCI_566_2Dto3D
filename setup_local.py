"""
setup_local.py — Download Objaverse GLBs, update dataset_split.json with local
paths, and render dataset images using ray casting (no OpenGL/display needed).

Run this once before training. Works on headless servers.

Usage:
  python setup_local.py --base-dir /path/to/project
"""

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Local setup for back-face fine-tuning")
    p.add_argument("--base-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                    help="Project root (contains TripoSR/, project_scripts/, etc.)")
    p.add_argument("--split-json", type=str, default=None)
    p.add_argument("--render-dir", type=str, default=None)
    p.add_argument("--render-size", type=int, default=512, help="Rendered image resolution")
    return p.parse_args()


def download_glbs(split_json):
    import objaverse

    with open(split_json) as f:
        split = json.load(f)

    all_uids = list({obj["uid"] for obj in split["hard_set"] + split["easy_set"]})
    print(f"Downloading {len(all_uids)} Objaverse objects...")
    objects = objaverse.load_objects(uids=all_uids)
    print(f"Downloaded {len(objects)} objects.")

    for subset in ["hard_set", "easy_set", "all_scored"]:
        for obj in split[subset]:
            if obj["uid"] in objects:
                obj["glb_path"] = objects[obj["uid"]]

    with open(split_json, "w") as f:
        json.dump(split, f, indent=2)

    for label, subset in [("hard", "hard_set"), ("easy", "easy_set")]:
        missing = [o["uid"][:8] for o in split[subset] if not os.path.exists(o["glb_path"])]
        if missing:
            print(f"MISSING {label} GLBs: {missing}")
        else:
            print(f"All {label} set GLBs found.")

    return split


def render_dataset(split, render_dir, render_manifest_path, size):
    """Render front-view images using ray casting (no OpenGL/display needed)."""
    import trimesh
    import numpy as np
    from PIL import Image

    os.makedirs(os.path.join(render_dir, "hard"), exist_ok=True)
    os.makedirs(os.path.join(render_dir, "easy"), exist_ok=True)

    manifest = {"hard": {}, "easy": {}}
    H = W = size

    for subset, label in [("hard_set", "hard"), ("easy_set", "easy")]:
        for obj in split[subset]:
            uid = obj["uid"]
            glb_path = obj["glb_path"]
            out_path = os.path.join(render_dir, label, f"{uid}.png")

            if os.path.exists(out_path):
                manifest[label][uid] = out_path
                continue

            if not os.path.exists(glb_path):
                print(f"  SKIP {uid[:8]}: GLB not found")
                continue

            print(f"  Rendering {uid[:8]} ({label})...")
            try:
                scene_or_mesh = trimesh.load(glb_path)
                if isinstance(scene_or_mesh, trimesh.Scene):
                    meshes = [g for g in scene_or_mesh.geometry.values()
                              if isinstance(g, trimesh.Trimesh)]
                    if not meshes:
                        print(f"  SKIP {uid[:8]}: no meshes")
                        continue
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    mesh = scene_or_mesh

                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
                mesh.apply_translation(-mesh.centroid)
                ext = max(mesh.extents)
                if ext > 0:
                    mesh.apply_scale(1.0 / ext)

                cam_dist = 1.9
                fov_rad = np.radians(40.0)
                half_h = cam_dist * np.tan(fov_rad / 2)

                ys = np.linspace(half_h, -half_h, H)
                xs = np.linspace(-half_h, half_h, W)
                xx, yy = np.meshgrid(xs, ys)

                origins = np.zeros((H * W, 3), dtype=np.float64)
                origins[:, 0] = xx.ravel()
                origins[:, 1] = yy.ravel()
                origins[:, 2] = cam_dist

                directions = np.zeros((H * W, 3), dtype=np.float64)
                directions[:, 2] = -1.0

                intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
                locations, index_ray, index_tri = intersector.intersects_location(
                    ray_origins=origins, ray_directions=directions, multiple_hits=False)

                rgb = np.ones((H * W, 3), dtype=np.float32) * 255
                if len(locations) > 0:
                    face_normals = mesh.face_normals[index_tri]
                    light_dir = np.array([0.0, 0.0, 1.0])
                    diffuse = np.abs(face_normals @ light_dir).astype(np.float32)
                    diffuse = np.clip(0.3 + 0.7 * diffuse, 0.0, 1.0)
                    rgb[index_ray] = (diffuse[:, None] * 255).astype(np.float32)

                img = Image.fromarray(rgb.reshape(H, W, 3).astype(np.uint8), "RGB")
                rgba = img.convert("RGBA")
                alpha = np.ones((H, W), dtype=np.uint8) * 255
                if len(locations) > 0:
                    hit_mask = np.zeros(H * W, dtype=bool)
                    hit_mask[index_ray] = True
                    alpha[~hit_mask.reshape(H, W)] = 0
                else:
                    alpha[:] = 0
                rgba.putalpha(Image.fromarray(alpha))
                rgba.save(out_path)
                manifest[label][uid] = out_path
                print(f"  Saved -> {out_path}")

            except Exception as e:
                print(f"  ERROR {uid[:8]}: {e}")

    with open(render_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nRender manifest saved to {render_manifest_path}")
    print(f"Total renders: hard={len(manifest['hard'])}, easy={len(manifest['easy'])}")


def main():
    args = parse_args()
    base = args.base_dir
    data_dir = os.path.join(base, "project_scripts/project_scripts")

    split_json = args.split_json or os.path.join(data_dir, "dataset_split.json")
    render_dir = args.render_dir or os.path.join(base, "dataset_renders")
    render_manifest = os.path.join(data_dir, "render_manifest.json")

    print("=" * 60)
    print("Local Setup for Back-Face Fine-tuning")
    print("=" * 60)
    print(f"  Base dir:   {base}")
    print(f"  Split JSON: {split_json}")
    print(f"  Render dir: {render_dir}")

    print("\n[1/3] Downloading Objaverse GLBs...")
    split = download_glbs(split_json)

    print("\n[2/3] Rendering dataset images...")
    missing = []
    for subset, label in [("hard_set", "hard"), ("easy_set", "easy")]:
        for obj in split[subset]:
            if not os.path.exists(os.path.join(render_dir, label, f"{obj['uid']}.png")):
                missing.append(obj["uid"])
    if missing:
        print(f"  {len(missing)} renders missing — generating...")
        render_dataset(split, render_dir, render_manifest, args.render_size)
    else:
        print(f"  All renders found in {render_dir}")
        if not os.path.exists(render_manifest):
            manifest = {"hard": {}, "easy": {}}
            for subset, label in [("hard_set", "hard"), ("easy_set", "easy")]:
                for obj in split[subset]:
                    manifest[label][obj["uid"]] = os.path.join(
                        render_dir, label, f"{obj['uid']}.png")
            with open(render_manifest, "w") as f:
                json.dump(manifest, f, indent=2)

    print("\n[3/3] Verifying setup...")
    with open(split_json) as f:
        split = json.load(f)
    n_hard = sum(1 for o in split["hard_set"] if os.path.exists(o["glb_path"]))
    n_easy = sum(1 for o in split["easy_set"] if os.path.exists(o["glb_path"]))
    print(f"  GLBs:    hard={n_hard}/10, easy={n_easy}/10")

    n_renders = sum(1 for subset, label in [("hard_set", "hard"), ("easy_set", "easy")]
                    for o in split[subset]
                    if os.path.exists(os.path.join(render_dir, label, f"{o['uid']}.png")))
    print(f"  Renders: {n_renders}/20")

    print("\nSetup complete. Next: python finetune_backface_v2.py --base-dir", base)


if __name__ == "__main__":
    main()
