"""
main.py — Run the full 3D reconstruction evaluation pipeline

Steps:
  1.  Filter Objaverse dataset into hard/easy split
  2.  Render 8 input views per GT object (flat shading)

  --- Baseline (TripoSR, no finetuning) ---
  3.  Run TripoSR baseline inference
  4.  Compute baseline metrics (CD, F-score, PSNR, LPIPS, PSNR-N, LPIPS-N)
  5.  Render baseline predictions for visual inspection

  --- Depth Supervision Finetuned ---
  6.  Depth supervision fine-tuning
  7.  Run finetuned inference
  8.  Compute finetuned metrics
  9.  Render finetuned predictions for visual inspection
"""
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    # ── Shared setup ──────────────────────────────────────────────────────────
    (
        "1. Filter Objaverse Dataset",
        "filter_objaverse.py",
        "dataset_split.json",
    ),
    (
        "2. Render 8 Input Views (GT)",
        "render_input_views.py",
        "input_view_manifest.json",
    ),

    # ── Baseline ──────────────────────────────────────────────────────────────
    (
        "3. Baseline Inference (TripoSR)",
        "run_inference.py",
        "inference_manifest.json",
    ),
    (
        "4. Baseline Metrics",
        "compute_metrics.py",
        "metrics_baseline.json",
    ),
    (
        "5. Render Baseline Predictions",
        "render_predictions.py",
        "prediction_renders",          # folder — exists after first run
    ),

    # ── Depth Supervision Finetuned ───────────────────────────────────────────
    (
        "6. Depth Supervision Fine-tuning",
        "finetune_depth.py",
        "checkpoints_depth/final.ckpt",
    ),
    (
        "7. Finetuned Inference",
        "run_inference_finetuned.py",
        "inference_manifest_depth.json",
    ),
    (
        "8. Finetuned Metrics",
        "compute_metrics_finetuned.py",
        "metrics_depth.json",
    ),
    (
        "9. Render Finetuned Predictions",
        "render_predictions_finetuned.py",
        "prediction_renders_depth",    # separate folder from baseline
    ),
]

for name, script, output in STEPS:
    print(f"\n{'='*50}\n{name}\n{'='*50}")

    output_path = os.path.join(SCRIPTS_DIR, output)
    if os.path.exists(output_path):
        print(f"  SKIP: output already exists → {output}")
        continue

    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, script)],
        cwd=SCRIPTS_DIR,
    )

    if result.returncode != 0:
        print(f"\nFailed at: {name}")
        sys.exit(1)

print("\nPipeline complete.")
print(f"\nBaseline:")
print(f"  Metrics           → {os.path.join(SCRIPTS_DIR, 'metrics_baseline.json')}")
print(f"  Predicted renders → {os.path.join(SCRIPTS_DIR, 'prediction_renders')}")
print(f"\nDepth Finetuned:")
print(f"  Metrics           → {os.path.join(SCRIPTS_DIR, 'metrics_depth.json')}")
print(f"  Predicted renders → {os.path.join(SCRIPTS_DIR, 'prediction_renders_depth')}")
print(f"\nShared:")
print(f"  GT renders        → {os.path.join(SCRIPTS_DIR, 'gt_renders')}")