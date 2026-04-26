import objaverse
import trimesh
import numpy as np
from PIL import Image
import pyrender
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

uids = objaverse.load_uids()
sample_uids = uids[:1]
objects = objaverse.load_objects(uids=sample_uids)
glb_path = list(objects.values())[0]
print(f"Downloaded to: {glb_path}")

# Load and center
mesh = trimesh.load(glb_path)
if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)

# normalize
mesh.apply_translation(-mesh.centroid)
scale = 1.0 / mesh.scale
mesh.apply_scale(scale)

py_mesh = pyrender.Mesh.from_trimesh(mesh)
scene = pyrender.Scene(ambient_light=np.array([0.3, 0.3, 0.3]))
scene.add(py_mesh)

extent = np.max(mesh.bounds[1] - mesh.bounds[0])
cam_distance = extent * 2.5

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
camera_pose = np.array([
    [1, 0,  0, 0],
    [0, 1,  0, 0],
    [0, 0,  1, cam_distance],
    [0, 0,  0, 1],
], dtype=np.float32)
scene.add(camera, pose=camera_pose)

light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
scene.add(light, pose=camera_pose)

r = pyrender.OffscreenRenderer(512, 512)
color, _ = r.render(scene)
r.delete()

Image.fromarray(color).save("reference.png")
print("Saved reference.png")
