import json
import os
import sys
import torch
import torch.nn.functional as F
import trimesh
import numpy as np
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

sys.path.insert(0, os.path.expanduser("~/TripoSR"))

from tsr.system import TSR
from tsr.utils import get_spherical_cameras
from tsr.utils import remove_background, resize_foreground, ImagePreprocessor

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SPLIT_JSON      = os.path.join(SCRIPT_DIR, "dataset_split.json")
RENDER_DIR      = os.path.join(SCRIPT_DIR, "dataset_renders")
CHECKPOINT_DIR  = os.path.join(SCRIPT_DIR, "checkpoints_depth")
LOG_FILE        = os.path.join(SCRIPT_DIR, "finetune_depth_log.json")

LAMBDA_DEPTH    = 0.1
LR              = 1e-5
N_EPOCHS        = 10
N_VIEWS         = 2
IMG_SIZE        = 32
SAVE_EVERY      = 2
DEVICE          = "cuda"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_and_normalize_mesh(glb_path):
    mesh = trimesh.load(glb_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))
    return mesh


def render_gt_rgb(mesh, rays_o, resolution):
    py_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(ambient_light=np.array([0.5, 0.5, 0.5]))
    scene.add(py_mesh)

    cam_pos = rays_o[resolution // 2, resolution // 2].cpu().numpy()

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(40.0), aspectRatio=1.0)
    camera_pose = np.array([
        [1, 0, 0, cam_pos[0]],
        [0, 1, 0, cam_pos[1]],
        [0, 0, 1, cam_pos[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(resolution, resolution)
    color, _ = r.render(scene)
    r.delete()

    return torch.from_numpy(color.astype(np.float32) / 255.0).to(DEVICE)


def render_gt_depth(mesh, rays_o, rays_d):
    rays_o_np = rays_o.reshape(-1, 3).cpu().numpy()
    rays_d_np = rays_d.reshape(-1, 3).cpu().numpy()

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, _ = intersector.intersects_location(
        ray_origins=rays_o_np,
        ray_directions=rays_d_np,
        multiple_hits=False
    )

    depth = np.zeros(len(rays_o_np), dtype=np.float32)
    if len(locations) > 0:
        diffs = locations - rays_o_np[index_ray]
        depths = np.linalg.norm(diffs, axis=-1)
        depth[index_ray] = depths

    depth_tensor = torch.from_numpy(depth).to(DEVICE)
    depth_tensor = depth_tensor.view(*rays_o.shape[:-1])
    return depth_tensor


def rgba_to_rgb_white_bg(image):
    if image.mode != "RGBA":
        return image.convert("RGB")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    return bg.convert("RGB")


def load_training_data(split, render_dir):
    data = []
    for obj in split["hard_set"]:
        uid      = obj["uid"]
        glb_path = obj["glb_path"]
        img_path = os.path.join(render_dir, "hard", f"{uid}.png")
        if os.path.exists(img_path) and os.path.exists(glb_path):
            data.append({"uid": uid, "glb_path": glb_path, "img_path": img_path})
    return data


def main():
    print("Loading TripoSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(0)
    model.to(DEVICE)
    model.train()

    for param in model.image_tokenizer.parameters():
        param.requires_grad = False
    for param in model.tokenizer.parameters():
        param.requires_grad = True
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.post_processor.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    with open(SPLIT_JSON) as f:
        split = json.load(f)

    train_data = load_training_data(split, RENDER_DIR)
    print(f"Training on {len(train_data)} objects (hard set)\n")

    log = []

    for epoch in range(1, N_EPOCHS + 1):
        epoch_loss_rgb   = []
        epoch_loss_depth = []
        epoch_loss_total = []

        for obj in train_data:
            uid      = obj["uid"]
            glb_path = obj["glb_path"]
            img_path = obj["img_path"]

            gt_mesh = load_and_normalize_mesh(glb_path)

            image = Image.open(img_path).convert("RGBA")
            image = remove_background(image)
            image = resize_foreground(image, 0.85)
            image = rgba_to_rgb_white_bg(image)

            optimizer.zero_grad()
            scene_codes = model([image], device=DEVICE)

            rays_o, rays_d = get_spherical_cameras(
                n_views=N_VIEWS,
                elevation_deg=0.0,
                camera_distance=1.9,
                fovy_deg=40.0,
                height=IMG_SIZE,
                width=IMG_SIZE,
            )
            rays_o = rays_o.to(DEVICE)
            rays_d = rays_d.to(DEVICE)

            loss_rgb_views   = []
            loss_depth_views = []

            for v in range(N_VIEWS):
                pred_rgb, pred_depth = model.renderer(
                    model.decoder,
                    scene_codes[0],
                    rays_o[v],
                    rays_d[v],
                )

                gt_rgb = render_gt_rgb(gt_mesh, rays_o[v], IMG_SIZE)

                gt_depth = render_gt_depth(gt_mesh, rays_o[v], rays_d[v])

                l_rgb = F.l1_loss(pred_rgb, gt_rgb)

                valid_depth_mask = gt_depth > 0
                if valid_depth_mask.sum() > 0:
                    l_depth = F.l1_loss(
                        pred_depth[valid_depth_mask],
                        gt_depth[valid_depth_mask]
                    )
                else:
                    l_depth = torch.tensor(0.0, device=DEVICE)

                loss_rgb_views.append(l_rgb)
                loss_depth_views.append(l_depth)

            l_rgb_mean   = torch.stack(loss_rgb_views).mean()
            l_depth_mean = torch.stack(loss_depth_views).mean()

            l_total = l_rgb_mean + LAMBDA_DEPTH * l_depth_mean

            l_total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            epoch_loss_rgb.append(l_rgb_mean.item())
            epoch_loss_depth.append(l_depth_mean.item())
            epoch_loss_total.append(l_total.item())

            print(f"  [{uid[:8]}] L_rgb={l_rgb_mean.item():.4f}  "
                  f"L_depth={l_depth_mean.item():.4f}  "
                  f"L_total={l_total.item():.4f}")

        scheduler.step()

        mean_rgb   = float(np.mean(epoch_loss_rgb))
        mean_depth = float(np.mean(epoch_loss_depth))
        mean_total = float(np.mean(epoch_loss_total))

        print(f"\nEpoch {epoch}/{N_EPOCHS} — "
              f"L_rgb={mean_rgb:.4f}  "
              f"L_depth={mean_depth:.4f}  "
              f"L_total={mean_total:.4f}\n")

        log.append({
            "epoch": epoch,
            "mean_rgb": mean_rgb,
            "mean_depth": mean_depth,
            "mean_total": mean_total,
        })

        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.ckpt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint → {ckpt_path}\n")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final.ckpt"))
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete. Final checkpoint and log saved.")


if __name__ == "__main__":
    main()
