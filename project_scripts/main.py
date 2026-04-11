"""
main.py — Run the full 3D reconstruction pipeline
"""

import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("1. Filter Objaverse Dataset",         "filter_objaverse.py",        "dataset_split.json"),
    ("2. Render Reference Images",          "render_dataset.py",           "render_manifest.json"),
    ("3. Baseline Inference",               "run_inference.py",            "inference_manifest.json"),
    ("4. Baseline Metrics",                 "compute_metrics.py",          "metrics_baseline.json"),
    ("5. Depth Supervision Fine-tuning",    "finetune_depth.py",           "checkpoints_depth/final.ckpt"),
    ("6. Fine-tuned Inference",             "run_inference_finetuned.py",  "inference_manifest_depth.json"),
    ("7. Fine-tuned Metrics + Comparison",  "compute_metrics_finetuned.py","metrics_depth.json"),
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
