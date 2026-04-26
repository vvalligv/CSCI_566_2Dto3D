"""
render_input_views.py
Renders 8 flat-shaded views of each GT GLB.
These images are fed into TripoSR for reconstruction.
Also saves the manifest: input_view_manifest.json
  {
    "hard": { "<uid>": ["view00.png", ..., "view07.png"] },
    "easy": { ... }
  }
"""
import json
import os
import trimesh
import numpy as np
from PIL import Image
import pyrender

os.environ["PYOPENGL_PLATFORM"] = "egl"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "input_views")
SPLIT_JSON = os.path.join(SCRIPT_DIR, "dataset_split.json")

RESOLUTION = 512
N_VIEWS    = 8
CAM_DIST   = 1.8
YFOV       = np.pi / 3.0

os.makedirs(os.path.join(OUTPUT_DIR, "hard"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "easy"), exist_ok=True)


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


# Shared camera poses — imported by inference and scoring scripts
CAMERA_POSES = get_camera_poses()


def normalize_mesh(mesh):
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))
    return mesh


def render_views(glb_path, uid, out_dir):
    try:
        mesh = trimesh.load(glb_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if mesh is None or len(mesh.faces) == 0:
            print(f"  SKIP {uid[:8]}: empty mesh")
            return []

        mesh = normalize_mesh(mesh)
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        r = pyrender.OffscreenRenderer(RESOLUTION, RESOLUTION)
        saved = []

        for i, cam_pose in enumerate(CAMERA_POSES):
            scene = pyrender.Scene(
                ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
                bg_color=np.array([0.5, 0.5, 0.5, 1.0]),
            )
            scene.add(py_mesh)
            scene.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0), pose=cam_pose)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)
            path = os.path.join(out_dir, f"{uid}_view{i:02d}.png")
            Image.fromarray(color).save(path)
            saved.append(path)

        r.delete()
        return saved

    except Exception as e:
        print(f"  ERROR {uid[:8]}: {e}")
        return []


def main():
    with open(SPLIT_JSON) as f:
        split = json.load(f)

    manifest = {"hard": {}, "easy": {}}

    for subset in ["hard_set", "easy_set"]:
        label   = "hard" if subset == "hard_set" else "easy"
        out_dir = os.path.join(OUTPUT_DIR, label)
        print(f"\nRendering input views for {label} set ({len(split[subset])} objects)...")

        for obj in split[subset]:
            uid      = obj["uid"]
            glb_path = obj["glb_path"]

            expected = [os.path.join(out_dir, f"{uid}_view{i:02d}.png") for i in range(N_VIEWS)]
            if all(os.path.exists(p) for p in expected):
                print(f"  CACHED {uid[:8]}")
                manifest[label][uid] = expected
                continue

            print(f"  Rendering {uid[:8]}...")
            saved = render_views(glb_path, uid, out_dir)
            if saved:
                manifest[label][uid] = saved
                print(f"  Saved {len(saved)} views")

    out_path = os.path.join(SCRIPT_DIR, "input_view_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(len(v) for v in manifest.values())
    print(f"\nDone. {total}/20 objects rendered.")
    print(f"Manifest → {out_path}")


if __name__ == "__main__":
    main()
