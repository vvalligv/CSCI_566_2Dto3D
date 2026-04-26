import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_JSON  = os.path.join(SCRIPT_DIR, "metrics_baseline.json")
DEPTH_JSON = os.path.join(SCRIPT_DIR, "metrics_depth.json")

# Run compute_metrics_finetuned.py if metrics_depth.json doesn't exist yet
if not os.path.exists(DEPTH_JSON):
    print("metrics_depth.json not found — running compute_metrics_finetuned.py...")
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "compute_metrics_finetuned.py")],
        cwd=SCRIPT_DIR,
    )
    if result.returncode != 0:
        print("compute_metrics_finetuned.py failed.")
        sys.exit(1)

def print_summary(data, title):
    s = data['summary']
    print('=' * 50)
    print(title)
    print('=' * 50)
    for label in ['hard', 'easy']:
        n  = s[label]['n']
        cd = s[label]['mean_chamfer']
        fs = s[label]['mean_f_score']
        ps = s[label].get('mean_psnr')
        lp = s[label].get('mean_lpips')
        print(f"{label.capitalize()} Set ({n} objects):")
        print(f"  Chamfer Distance : {cd:.4f}")
        print(f"  F-score          : {fs:.4f}")
        if ps is not None:
            print(f"  PSNR             : {ps:.2f} dB")
        if lp is not None:
            print(f"  LPIPS            : {lp:.4f}")
    print()
    print('Per-object breakdown:')
    for label in ['hard', 'easy']:
        print(f'  {label.upper()}:')
        for obj in data['per_object'][label]:
            cd       = obj.get('chamfer_distance') or obj.get('mean_chamfer')
            fs       = obj.get('f_score')          or obj.get('mean_f_score')
            cd_delta = obj.get('cd_delta')
            fs_delta = obj.get('fs_delta')
            delta_cd = f"{cd_delta:+.4f}" if cd_delta is not None else 'N/A'
            delta_fs = f"{fs_delta:+.4f}" if fs_delta is not None else 'N/A'
            cd_str   = f"{cd:.4f}" if cd is not None else 'N/A'
            fs_str   = f"{fs:.4f}" if fs is not None else 'N/A'
            print(f"    {obj['uid'][:8]}  CD={cd_str} (Δ{delta_cd})  F={fs_str} (Δ{delta_fs})")

print_summary(json.load(open(BASE_JSON)),  'RESULTS BASELINE SUMMARY')
print()
print_summary(json.load(open(DEPTH_JSON)), 'RESULTS DEPTH SUMMARY')