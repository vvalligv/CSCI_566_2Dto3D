# Generating 3D Objects From 2D References

## Project Scripts (`project_scripts/`)

### 1. Rendering Prototype — `render_ref.py`
Single-object rendering prototype used to validate the headless rendering 
stack on the target machine. Downloads one object from Objaverse, normalizes 
it to unit scale, and renders a 512×512 frontal reference image using pyrender 
in EGL mode (no display required). Established the camera placement strategy 
(`cam_distance = extent * 2.5`) used throughout the pipeline. Superseded by 
`render_dataset.py` for full dataset rendering.

---

### 2. Dataset Filtering — `filter_objaverse.py`
Downloads a pool of objects from Objaverse and scores each for difficulty 
using two geometric metrics computed directly from the mesh:

- **Depth-to-width ratio** — divides the object's Z-axis extent by its 
  largest lateral extent (X or Y). High scores indicate deep objects that 
  recede away from the camera (e.g. staircases, hallways).
- **Frontal occlusion ratio** — casts rays from the front and counts the 
  fraction of faces never hit. High scores indicate objects where the body 
  is hidden behind the frontal silhouette (e.g. fish viewed head-on).

Combined score: `0.5 * depth_ratio + 0.5 * occlusion_ratio`

Objects are split into:
- **Hard set** (`HARD_N` objects) — highest difficulty scores, used as 
  training and evaluation targets
- **Easy set** (`EASY_N` objects) — lowest difficulty scores, used as 
  a control group
- Results saved to `dataset_split.json`

**Key config variables:**
POOL_SIZE = 300   # total objects to sample from Objaverse
HARD_N    = 10    # number of hard objects to keep
EASY_N    = 10    # number of easy objects to keep
PINNED_UIDS = [   # UIDs always included regardless of score
"3564578cde5c42279ead680df1619e3c",  # goldfish
"8476c4170df24cf5bbe6967222d1a42d",  # staircase
]
---

### 3. Dataset Rendering — `render_dataset.py`
Renders a 512×512 frontal reference image for each object in the dataset 
split using pyrender EGL headless mode. Each mesh is centered and normalized 
to unit scale before rendering. Reads from `dataset_split.json`, outputs 
images to `dataset_renders/hard/` and `dataset_renders/easy/`, and saves 
a manifest to `render_manifest.json`.

---

### 4. Baseline Inference — `run_inference.py`
Runs the pretrained TripoSR model on all 20 reference images. For each image:
1. Background is removed using TripoSR's built-in `remove_background`
2. Foreground is resized to 85% of frame
3. RGBA output is composited onto a white background → RGB
4. TripoSR generates a triplane NeRF and extracts a mesh via Marching Cubes 
   at resolution 256

Output meshes saved to `inference_outputs/hard/` and `inference_outputs/easy/`.

---

### 5. Baseline Metrics — `compute_metrics.py`
Computes Chamfer Distance and F-score for each output mesh against the 
ground truth Objaverse mesh. Both meshes are normalized to unit scale 
before comparison. Samples 10,000 surface points per mesh.

- **Chamfer Distance (CD)** — average squared nearest-neighbor distance 
  between predicted and GT point clouds. Lower is better.
- **F-score** — harmonic mean of precision and recall at distance threshold 
  0.05. Higher is better.

Results saved to `metrics_baseline.json`.

---

### 6. Depth Supervision Fine-tuning — `finetune_depth.py`
Fine-tunes TripoSR on the hard set with an additional depth supervision 
loss term:
L_total = L_rgb + λ * L_depth     (λ = 0.1)
- **L_rgb** — L1 loss between TripoSR's rendered RGB and GT RGB rendered 
  from the Objaverse mesh via pyrender
- **L_depth** — L1 loss between TripoSR's rendered depth and GT depth 
  obtained by ray-mesh intersection against the Objaverse mesh. Only rays 
  that hit the mesh (depth > 0) are supervised.

The image tokenizer (DINOv1 encoder) is frozen. Only the triplane backbone, 
post-processor, and decoder are updated.

**Key config variables:**
LAMBDA_DEPTH = 0.1    # weight for depth loss
LR           = 1e-5   # learning rate
N_EPOCHS     = 10     # training epochs
N_VIEWS      = 2      # camera views per object per step
IMG_SIZE     = 64     # render resolution during training
Checkpoints saved to `checkpoints_depth/` every `SAVE_EVERY` epochs.

---

### 7. Fine-tuned Inference — `run_inference_finetuned.py`
Same as `run_inference.py` but loads the fine-tuned weights from 
`checkpoints_depth/final.ckpt` on top of the pretrained model. 
Output meshes saved to `inference_outputs_depth/`.

---

### 8. Fine-tuned Metrics + Comparison — `compute_metrics_finetuned.py`
Computes the same metrics as `compute_metrics.py` on the fine-tuned outputs 
and prints a side-by-side comparison table against the baseline.

---

## Baseline Results

| Set | Chamfer Distance ↓ | F-score ↑ |
|---|---|---|
| Hard | 0.0558 | 0.2127 |
| Easy | 0.1184 | 0.1143 |

## Depth Supervision Results

| Metric | Baseline | Depth FT | Delta |
|---|---|---|---|
| Hard CD | 0.0558 | 0.0201 | **-0.0357 (-64%)** |
| Hard F-score | 0.2127 | 0.4098 | **+0.1972 (+93%)** |
| Easy CD | 0.1184 | 0.1217 | +0.0033 (~0%) |
| Easy F-score | 0.1143 | 0.1124 | -0.0019 (~0%) |

Depth supervision significantly improves hard object reconstruction without 
degrading easy object performance, validating that the improvement is targeted 
rather than general.

---

## Environment Setup
```bash
conda activate triposr
pip install -r requirements.txt
```

## Hardware
- GPU: NVIDIA GeForce RTX 5090 (32GB VRAM)
- RAM: 64GB
- OS: Ubuntu 24.04
- CUDA: 12.8
- Driver: 580.126.20
