"""
finetune_backface_v2.py — Fine-tune TripoSR with back-face consistency loss.

Improvements over v1:
  1. Silhouette loss prevents density collapse during fine-tuning
  2. Discriminative LR (backbone 100x lower than decoder)
  3. Back-face loss warm-up over first N epochs
  4. Lower back-face weight (0.05 vs 0.1)
  5. Trains on both hard + easy sets

Usage:
  python finetune_backface_v2.py --base-dir /path/to/project
"""

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description="Back-face fine-tuning v2 for TripoSR")
    p.add_argument("--base-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                    help="Project root (contains TripoSR/, dataset_renders/, etc.)")
    p.add_argument("--split-json", type=str, default=None,
                    help="Path to dataset_split.json (default: <base-dir>/project_scripts/project_scripts/dataset_split.json)")
    p.add_argument("--render-dir", type=str, default=None,
                    help="Directory with rendered dataset images (default: <base-dir>/dataset_renders)")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Where to save checkpoints (default: <base-dir>/checkpoints_backface_v2)")
    p.add_argument("--lr", type=float, default=5e-5,
                    help="Learning rate for decoder/post-processor (backbone gets LR*0.01)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=3,
                    help="Linearly ramp back-face loss over this many epochs")
    p.add_argument("--lambda-backface", type=float, default=0.05)
    p.add_argument("--lambda-sil", type=float, default=1.0,
                    help="Silhouette loss weight (prevents density collapse)")
    p.add_argument("--n-views", type=int, default=2, help="Number of front views per object")
    p.add_argument("--img-size", type=int, default=64, help="Rendered image resolution")
    p.add_argument("--save-every", type=int, default=2, help="Save checkpoint every N epochs")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ─── Mesh Loading ─────────────────────────────────────────────────────────────

def load_and_normalize_mesh(glb_path):
    scene_or_mesh = trimesh.load(glb_path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No meshes in {glb_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    mesh.apply_translation(-mesh.centroid)
    ext = max(mesh.extents)
    if ext < 1e-8:
        raise ValueError(f"Degenerate mesh in {glb_path}")
    mesh.apply_scale(1.0 / ext)
    return mesh


# ─── GT Rendering (ray casting, no pyrender) ──────────────────────────────────

def render_gt_rgb_raycasting(mesh, rays_o, rays_d, device):
    H, W = rays_o.shape[:2]
    rays_o_np = rays_o.reshape(-1, 3).cpu().numpy().astype(np.float64)
    rays_d_np = rays_d.reshape(-1, 3).cpu().numpy().astype(np.float64)

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=rays_o_np, ray_directions=rays_d_np, multiple_hits=False)

    rgb = np.ones((H * W, 3), dtype=np.float32)
    if len(locations) > 0:
        face_normals = mesh.face_normals[index_tri]
        light_dir = np.array([0.0, 0.0, 1.0])
        diffuse = np.abs(face_normals @ light_dir).astype(np.float32)
        diffuse = np.clip(0.3 + 0.7 * diffuse, 0.0, 1.0)
        rgb[index_ray] = diffuse[:, None]

    return torch.from_numpy(rgb.reshape(H, W, 3)).to(device)


def render_gt_silhouette(mesh, rays_o, rays_d, device):
    """Ray-cast binary silhouette: 1 where mesh is hit, 0 where miss."""
    H, W = rays_o.shape[:2]
    rays_o_np = rays_o.reshape(-1, 3).cpu().numpy().astype(np.float64)
    rays_d_np = rays_d.reshape(-1, 3).cpu().numpy().astype(np.float64)

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    hits = intersector.intersects_any(ray_origins=rays_o_np, ray_directions=rays_d_np)
    return torch.from_numpy(hits.astype(np.float32)).reshape(H, W).to(device)


# ─── Back-Face Camera ─────────────────────────────────────────────────────────

def get_backface_cameras(get_spherical_cameras_fn, camera_distance=1.9,
                         fovy_deg=40.0, height=64, width=64):
    rays_o, rays_d = get_spherical_cameras_fn(
        n_views=1, elevation_deg=0.0, camera_distance=camera_distance,
        fovy_deg=fovy_deg, height=height, width=width)
    rot = torch.tensor([[-1., 0., 0.],
                        [ 0., 1., 0.],
                        [ 0., 0.,-1.]], dtype=rays_o.dtype)
    s = rays_o.shape
    return ((rays_o.reshape(-1, 3) @ rot.T).reshape(s),
            (rays_d.reshape(-1, 3) @ rot.T).reshape(s))


# ─── Image Preprocessing ──────────────────────────────────────────────────────

def rgba_to_rgb_white_bg(image):
    if image.mode != "RGBA":
        return image.convert("RGB")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    return bg.convert("RGB")


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_training_data(split, render_dir):
    """Load from both hard and easy sets."""
    data = []
    for subset, label in [("hard_set", "hard"), ("easy_set", "easy")]:
        for obj in split[subset]:
            uid = obj["uid"]
            glb_path = obj["glb_path"]
            img_path = os.path.join(render_dir, label, f"{uid}.png")
            if os.path.exists(img_path) and os.path.exists(glb_path):
                data.append({"uid": uid, "glb_path": glb_path,
                             "img_path": img_path, "label": label})
            else:
                print(f"  SKIP {uid[:8]}: img={os.path.exists(img_path)} "
                      f"glb={os.path.exists(glb_path)}")
    return data


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    base = args.base_dir
    split_json = args.split_json or os.path.join(base, "project_scripts/project_scripts/dataset_split.json")
    render_dir = args.render_dir or os.path.join(base, "dataset_renders")
    ckpt_dir = args.checkpoint_dir or os.path.join(base, "checkpoints_backface_v2")
    log_file = os.path.join(ckpt_dir, "training_log.json")

    sys.path.insert(0, os.path.join(base, "TripoSR"))
    from tsr.system import TSR
    from tsr.utils import get_spherical_cameras, remove_background, resize_foreground

    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 60)
    print("Back-Face Fine-tuning v2")
    print("=" * 60)
    print(f"  Base dir:       {base}")
    print(f"  Split JSON:     {split_json}")
    print(f"  Render dir:     {render_dir}")
    print(f"  Checkpoint dir: {ckpt_dir}")

    print("\nLoading TripoSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(0)
    model.to(args.device)
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

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_train:,}")

    param_groups = [
        {"params": list(model.backbone.parameters()), "lr": args.lr * 0.01},
        {"params": list(model.tokenizer.parameters()), "lr": args.lr * 0.01},
        {"params": list(model.post_processor.parameters()), "lr": args.lr},
        {"params": list(model.decoder.parameters()), "lr": args.lr},
    ]
    optimizer = Adam(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    with open(split_json) as f:
        split = json.load(f)

    train_data = load_training_data(split, render_dir)
    print(f"Training on {len(train_data)} objects (hard + easy)")
    print(f"Loss = L_rgb + {args.lambda_sil}*L_sil + ramp*{args.lambda_backface}*L_backface")
    print(f"Back-face warm-up over {args.warmup_epochs} epochs")
    print(f"LR = {args.lr} (backbone: {args.lr * 0.01})\n")

    if len(train_data) == 0:
        print("ERROR: No training data found. Run setup_local.py first.")
        return

    log = []

    for epoch in range(1, args.epochs + 1):
        ramp = min(1.0, epoch / args.warmup_epochs)
        e_rgb, e_sil, e_back, e_total = [], [], [], []

        for obj in train_data:
            uid, glb_path, img_path = obj["uid"], obj["glb_path"], obj["img_path"]

            try:
                gt_mesh = load_and_normalize_mesh(glb_path)
            except Exception as e:
                print(f"  SKIP {uid[:8]}: {e}")
                continue

            image = Image.open(img_path).convert("RGBA")
            image = remove_background(image)
            image = resize_foreground(image, 0.85)
            image = rgba_to_rgb_white_bg(image)

            optimizer.zero_grad()
            scene_codes = model([image], device=args.device)

            rays_o, rays_d = get_spherical_cameras(
                n_views=args.n_views, elevation_deg=0.0, camera_distance=1.9,
                fovy_deg=40.0, height=args.img_size, width=args.img_size)
            rays_o = rays_o.to(args.device)
            rays_d = rays_d.to(args.device)

            loss_rgb_views, loss_sil_views = [], []
            for v in range(args.n_views):
                pred_rgb, pred_opacity = model.renderer(
                    model.decoder, scene_codes[0], rays_o[v], rays_d[v])
                gt_rgb = render_gt_rgb_raycasting(gt_mesh, rays_o[v], rays_d[v], args.device)
                gt_sil = render_gt_silhouette(gt_mesh, rays_o[v], rays_d[v], args.device)
                loss_rgb_views.append(F.l1_loss(pred_rgb, gt_rgb))
                pred_mask = pred_opacity.squeeze(-1) if pred_opacity.dim() > 2 else pred_opacity
                loss_sil_views.append(F.binary_cross_entropy(
                    pred_mask.clamp(1e-6, 1 - 1e-6), gt_sil))

            l_rgb = torch.stack(loss_rgb_views).mean()
            l_sil = torch.stack(loss_sil_views).mean()

            rays_o_back, rays_d_back = get_backface_cameras(
                get_spherical_cameras, camera_distance=1.9,
                fovy_deg=40.0, height=args.img_size, width=args.img_size)
            rays_o_back = rays_o_back.to(args.device)
            rays_d_back = rays_d_back.to(args.device)

            pred_rgb_back, _ = model.renderer(
                model.decoder, scene_codes[0], rays_o_back[0], rays_d_back[0])
            gt_rgb_back = render_gt_rgb_raycasting(
                gt_mesh, rays_o_back[0], rays_d_back[0], args.device)
            l_backface = F.l1_loss(pred_rgb_back, gt_rgb_back)

            l_total = l_rgb + args.lambda_sil * l_sil + ramp * args.lambda_backface * l_backface
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            e_rgb.append(l_rgb.item())
            e_sil.append(l_sil.item())
            e_back.append(l_backface.item())
            e_total.append(l_total.item())

            print(f"  [{uid[:8]}|{obj['label']}] "
                  f"rgb={l_rgb.item():.4f}  sil={l_sil.item():.4f}  "
                  f"bf={l_backface.item():.4f}  total={l_total.item():.4f}")

        scheduler.step()

        mr, ms, mb, mt = (float(np.mean(x)) if x else 0.0
                          for x in [e_rgb, e_sil, e_back, e_total])
        print(f"\nEpoch {epoch}/{args.epochs} (ramp={ramp:.2f}) — "
              f"rgb={mr:.4f}  sil={ms:.4f}  bf={mb:.4f}  total={mt:.4f}\n")

        log.append({"epoch": epoch, "ramp": ramp,
                    "mean_rgb": mr, "mean_sil": ms,
                    "mean_backface": mb, "mean_total": mt})

        if epoch % args.save_every == 0:
            ckpt = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.ckpt")
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint -> {ckpt}\n")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.ckpt"))
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Done. Checkpoints -> {ckpt_dir}")


if __name__ == "__main__":
    main()
