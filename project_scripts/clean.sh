#!/bin/bash
cd ~/project_scripts

rm -f dataset_split.json
rm -f render_manifest.json
rm -f inference_manifest.json
rm -f inference_manifest_depth.json
rm -f metrics_baseline.json
rm -f metrics_depth.json
rm -f metrics_comparison.json
rm -f finetune_depth_log.json
rm -rf dataset_renders/
rm -rf inference_outputs/
rm -rf inference_outputs_depth/
rm -rf checkpoints_depth/

echo "All outputs cleared."
ls ~/project_scripts/
