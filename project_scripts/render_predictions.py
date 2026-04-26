"""
render_predictions.py
Renders each predicted mesh (with vertex colors) and GT mesh from all 8 camera poses.
Saves individual views and a contact sheet per object.

Output:
  prediction_renders/{hard,easy}/{uid}_inputview{j}_camview{i}.png
  prediction_renders/{hard,easy}/{uid}_inputview{j}_sheet.png
  gt_renders/{hard,easy}/{uid}_camview{i}.png
  gt_renders/{hard,easy}/{uid}_sheet.png
"""
import json
import os
import numpy as np
import trimesh
from PIL import Image
import pyrender

os.environ["PYOPENGL_PLATFORM"] = "egl"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SPLIT_JSON   = os.path.join(SCRIPT_DIR, "dataset_split.json")
INF_MANIFEST = os.path.join(SCRIPT_DIR, "inference_manifest.json")

PRED_OUT_DIR = os.path.join(SCRIPT_DIR, "prediction_renders")
GT_OUT_DIR   = os.path.join(SCRIPT_DIR, "gt_renders")

RESOLUTION = 512
N_VIEWS    = 8
CAM_DIST   = 1.8
YFOV       = np.pi / 3.0

for d in [PRED_OUT_DIR, GT_OUT_DIR]:
    os.makedirs(os.path.join(d, "hard"), exist_ok=True)
    os.makedirs(os.path.join(d, "easy"), exist_ok=True)


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


CAMERA_POSES = get_camera_poses()


def load_and_normalize(path):
    """Load mesh preserving vertex colors/materials, then normalize to unit size."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))
    return mesh


def render_views(mesh, out_dir, filename_prefix):
    """Render mesh from all N_VIEWS camera poses. Returns list of saved paths."""
    try:
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        r = pyrender.OffscreenRenderer(RESOLUTION, RESOLUTION)
        saved = []

        for i, cam_pose in enumerate(CAMERA_POSES):
            out_path = os.path.join(out_dir, f"{filename_prefix}_camview{i:02d}.png")
            if os.path.exists(out_path):
                saved.append(out_path)
                continue

            scene = pyrender.Scene(
                ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
                bg_color=np.array([0.5, 0.5, 0.5, 1.0]),
            )
            scene.add(py_mesh)
            scene.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0), pose=cam_pose)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)
            Image.fromarray(color).save(out_path)
            saved.append(out_path)

        r.delete()
        return saved

    except Exception as e:
        print(f"    Render error: {e}")
        return []


def make_contact_sheet(image_paths, out_path):
    """Stitch images into a single horizontal contact sheet."""
    images = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]
    if not images:
        return
    w, h  = images[0].size
    sheet = Image.new("RGB", (w * len(images), h), color=(30, 30, 30))
    for i, img in enumerate(images):
        sheet.paste(img, (i * w, 0))
    sheet.save(out_path)


def main():
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    with open(INF_MANIFEST) as f:
        inf_manifest = json.load(f)

    gt_paths = {}
    for subset in ["hard_set", "easy_set"]:
        for obj in split[subset]:
            gt_paths[obj["uid"]] = obj["glb_path"]

    for uid, info in inf_manifest.items():
        label      = info["label"]
        pred_paths = info["meshes"]
        gt_path    = gt_paths.get(uid)

        pred_out = os.path.join(PRED_OUT_DIR, label)
        gt_out   = os.path.join(GT_OUT_DIR,   label)

        print(f"\n{uid[:8]} ({label})")

        # ── Render GT ────────────────────────────────────────────────────────
        gt_sheet = os.path.join(gt_out, f"{uid}_sheet.png")
        if os.path.exists(gt_sheet):
            print(f"  GT → CACHED")
        elif gt_path and os.path.exists(gt_path):
            try:
                gt_mesh   = load_and_normalize(gt_path)
                gt_renders = render_views(gt_mesh, gt_out, uid)
                if gt_renders:
                    make_contact_sheet(gt_renders, gt_sheet)
                    print(f"  GT → {len(gt_renders)} views + sheet")
            except Exception as e:
                print(f"  ERROR rendering GT: {e}")

        # ── Render each predicted mesh ────────────────────────────────────────
        for j, pred_path in enumerate(pred_paths):
            if not os.path.exists(pred_path):
                print(f"  pred view{j:02d} SKIP: not found")
                continue

            sheet_path = os.path.join(pred_out, f"{uid}_inputview{j:02d}_sheet.png")
            if os.path.exists(sheet_path):
                print(f"  pred view{j:02d} → CACHED")
                continue

            try:
                pred_mesh = load_and_normalize(pred_path)
                prefix    = f"{uid}_inputview{j:02d}"
                renders   = render_views(pred_mesh, pred_out, prefix)
                if renders:
                    make_contact_sheet(renders, sheet_path)
                    print(f"  pred view{j:02d} → {len(renders)} renders + sheet")
            except Exception as e:
                print(f"  pred view{j:02d} ERROR: {e}")

    print(f"\nDone.")
    print(f"  Predicted renders → {PRED_OUT_DIR}")
    print(f"  GT renders        → {GT_OUT_DIR}")


if __name__ == "__main__":
    main()