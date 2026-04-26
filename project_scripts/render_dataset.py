import json
import os
import trimesh
import numpy as np
from PIL import Image
import pyrender

os.environ["PYOPENGL_PLATFORM"] = "egl"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset_renders")
SPLIT_JSON = os.path.join(SCRIPT_DIR, "dataset_split.json")
RESOLUTION = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "hard"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "easy"), exist_ok=True)

def render_object(glb_path, uid, output_path):
    try:
        mesh = trimesh.load(glb_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if mesh is None or len(mesh.faces) == 0:
            print(f"  SKIP {uid[:8]}: empty mesh")
            return False

        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(1.0 / mesh.scale)

        py_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(ambient_light=np.array([0.4, 0.4, 0.4]))
        scene.add(py_mesh)

        extent = np.max(mesh.bounds[1] - mesh.bounds[0])
        cam_distance = extent * 2.5

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, cam_distance],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        scene.add(light, pose=camera_pose)

        r = pyrender.OffscreenRenderer(RESOLUTION, RESOLUTION)
        color, _ = r.render(scene)
        r.delete()

        Image.fromarray(color).save(output_path)
        return True

    except Exception as e:
        print(f"  ERROR {uid[:8]}: {e}")
        return False


def main():
    with open(SPLIT_JSON) as f:
        split = json.load(f)

    results = {"hard": {}, "easy": {}}

    for subset in ["hard_set", "easy_set"]:
        label = "hard" if subset == "hard_set" else "easy"
        print(f"\nRendering {label} set ({len(split[subset])} objects)...")

        for obj in split[subset]:
            uid = obj["uid"]
            glb_path = obj["glb_path"]
            output_path = os.path.join(OUTPUT_DIR, label, f"{uid}.png")

            if os.path.exists(output_path):
                print(f"  CACHED {uid[:8]}...")
                results[label][uid] = output_path
                continue

            print(f"  Rendering {uid[:8]}... (depth_ratio={obj['depth_ratio']:.2f})")
            success = render_object(glb_path, uid, output_path)
            if success:
                results[label][uid] = output_path
                print(f"  Saved → {output_path}")

    manifest_path = os.path.join(SCRIPT_DIR, "render_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    total = sum(len(v) for v in results.values())
    print(f"\nDone. Rendered {total}/20 objects.")
    print(f"Manifest saved to {manifest_path}")
    print(f"Images saved to {OUTPUT_DIR}/hard/ and {OUTPUT_DIR}/easy/")


if __name__ == "__main__":
    main()
