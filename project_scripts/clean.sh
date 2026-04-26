#!/bin/bash
cd "$(dirname "$(realpath "$0")")"

rm -f dataset_split.json
rm -f render_manifest.json
rm -f inference_manifest.json
rm -f inference_manifest2.json
rm -f inference_manifest_depth.json
rm -f metrics_baseline.json
rm -f metrics_baseline2.json
rm -f metrics_depth.json
rm -f metrics_comparison.json
rm -f finetune_depth_log.json
rm -f input_view_manifest.json
rm -rf dataset_renders/
rm -rf inference_outputs/
rm -rf inference_outputs_depth/
rm -rf checkpoints_depth/
rm -rf input_views/
rm -rf prediction_renders/
rm -rf gt_renders/
rm -rf dataset_renders/

echo "All outputs cleared."
