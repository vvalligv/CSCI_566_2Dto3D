"""
main.py — Run the full 3D reconstruction evaluation pipeline

Steps:
  1. Filter Objaverse dataset into hard/easy split
  2. Render 8 input views per GT object (flat shading)
  3. Run TripoSR on each view → 8 predicted meshes per object
  4. Compute all metrics (CD, F-score, PSNR, LPIPS, PSNR-N, LPIPS-N)
  5. Render predicted meshes + GT for visual inspection
"""
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("1. Filter Objaverse Dataset",       "filter_objaverse.py",    "dataset_split.json"),
    ("2. Render 8 Input Views (GT)",      "render_input_views.py",  "input_view_manifest.json"),
    ("3. Run TripoSR Inference",          "run_inference2.py",       "inference_manifest2.json"),
    ("4. Compute All Metrics",            "compute_metrics2.py",     "metrics_baseline2.json"),
    ("5. Render Predictions for Review",  "render_predictions.py",  "prediction_renders"),
]

for name, script, output in STEPS:
    print(f"\n{'='*50}\n{name}\n{'='*50}")

    output_path = os.path.join(SCRIPTS_DIR, output)
    if os.path.exists(output_path):
        print(f"  SKIP: output already exists → {output}")
        continue

    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, script)],
        cwd=SCRIPTS_DIR
    )

    if result.returncode != 0:
        print(f"\nFailed at: {name}")
        sys.exit(1)

print("\nPipeline complete.")
print(f"  Metrics           → {os.path.join(SCRIPTS_DIR, 'metrics_baseline.json')}")
print(f"  Predicted renders → {os.path.join(SCRIPTS_DIR, 'prediction_renders')}")
print(f"  GT renders        → {os.path.join(SCRIPTS_DIR, 'gt_renders')}")