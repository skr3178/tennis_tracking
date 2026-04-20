#!/bin/bash
# Post-training evaluation script
# Waits for training to finish, then runs:
#   Stage 1: Synthetic test eval (NRMSE on held-out sequences)
#   Stage 2: Real video inference with rally segmentation + reprojection metrics
#
# Usage: nohup bash run_post_training_eval.sh <training_pid> > post_training_eval.log 2>&1 &

set -e

TRAIN_PID=${1:?Usage: $0 <training_pid>}
CKPT="checkpoints_trimmed/best.pt"
WASB_CSV="../wasb_ball_positions.csv"
CAMERA_JSON="inference_output/calibrated_camera.json"
OUT_DIR="inference_output/trimmed_eval"

cd "$(dirname "$0")"
source ../.venv/bin/activate

echo "=============================================="
echo "Post-training evaluation"
echo "Waiting for training PID $TRAIN_PID to finish..."
echo "=============================================="

# Wait for training to complete
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 10
done
echo "Training finished at $(date)"
echo ""

# Show final training metrics
echo "=============================================="
echo "Final training log entries:"
echo "=============================================="
grep "val NRMSE" training_trimmed.log | tail -5
echo ""

# ---- Stage 1: Synthetic test eval ----
echo "=============================================="
echo "STAGE 1: Synthetic Test Evaluation"
echo "=============================================="
python eval.py \
    --ckpt "$CKPT" \
    --split_subdir ours_game_1000_trimmed \
    --split test \
    --device cuda
echo ""

# Also eval on val split for comparison
echo "--- Val split ---"
python eval.py \
    --ckpt "$CKPT" \
    --split_subdir ours_game_1000_trimmed \
    --split val \
    --device cuda
echo ""

# Also eval on paper's synthetic data (cross-dataset generalization)
echo "--- Paper synthetic test ---"
python eval.py \
    --ckpt "$CKPT" \
    --split_subdir synthetic \
    --split test \
    --device cuda
echo ""

# ---- Stage 2: Real video inference ----
echo "=============================================="
echo "STAGE 2: Real Video Inference (Rally-Segmented)"
echo "=============================================="
python infer_video_segmented.py \
    --wasb_csv "$WASB_CSV" \
    --camera_json "$CAMERA_JSON" \
    --ckpt "$CKPT" \
    --out_dir "$OUT_DIR" \
    --device cuda
echo ""

# ---- Stage 2b: Reprojection error ----
echo "=============================================="
echo "STAGE 2b: Reprojection Error Analysis"
echo "=============================================="
python -c "
import json, csv, numpy as np, sys, os
sys.path.insert(0, os.path.abspath('..'))
from LSTM_Model.data.parameterization import _ray_dir_camera

# Load results
detail = json.load(open('$OUT_DIR/rally_segmented_detail.json'))
cam = detail['camera']

# Load camera params
intrinsics = np.array(cam['intrinsics'], dtype=np.float64)
E = np.array(cam['extrinsic'], dtype=np.float64)
fx, cx, cy = intrinsics
fy = fx  # square pixels

print(f'Camera: fx={fx:.0f}, cx={cx:.0f}, cy={cy:.0f}')
print(f'Model checkpoint epoch: {detail[\"model_epoch\"]}')
print(f'Number of rallies: {detail[\"num_rallies\"]}')
print()

# For each rally, project 3D predictions back to 2D and compare with input UV
total_reproj_err = []
total_frames = 0

for rally in detail['rallies']:
    xyz = np.array(rally['xyz'], dtype=np.float64)
    uv_input = np.array(rally['uv'], dtype=np.float64)

    # Project xyz -> pixel: p_cam = E @ [x,y,z,1]^T, then u = cx + fx * x_cam / (-z_cam)
    ones = np.ones((len(xyz), 1))
    homog = np.concatenate([xyz, ones], axis=1)  # (N, 4)
    p_cam = (E @ homog.T).T  # (N, 4) -> (N, 3) cam coords

    z_cam = p_cam[:, 2]
    u_proj = cx + fx * (p_cam[:, 0] / (-z_cam))
    v_proj = cy - fy * (p_cam[:, 1] / (-z_cam))

    uv_proj = np.stack([u_proj, v_proj], axis=1)

    # Reprojection error per frame
    err = np.sqrt(((uv_proj - uv_input) ** 2).sum(axis=1))
    total_reproj_err.extend(err.tolist())
    total_frames += len(err)

    print(f'Rally {rally[\"rally_idx\"]}: {len(xyz)} frames, '
          f'reproj err: mean={err.mean():.1f}px, median={np.median(err):.1f}px, '
          f'max={err.max():.1f}px, <5px={100*(err<5).mean():.0f}%')

all_err = np.array(total_reproj_err)
print(f'')
print(f'Overall ({total_frames} frames):')
print(f'  Mean reproj error:   {all_err.mean():.2f} px')
print(f'  Median reproj error: {np.median(all_err):.2f} px')
print(f'  Std reproj error:    {all_err.std():.2f} px')
print(f'  <5px:  {100*(all_err<5).mean():.1f}%')
print(f'  <10px: {100*(all_err<10).mean():.1f}%')
print(f'  <20px: {100*(all_err<20).mean():.1f}%')

# Physical plausibility checks
print()
print('Physical plausibility:')
for rally in detail['rallies']:
    xyz = np.array(rally['xyz'])
    heights = xyz[:, 1]
    # Check: max height reasonable (< 8m for tennis)
    # Check: heights mostly positive (ball above ground)
    # Check: trajectory length reasonable
    neg_h = (heights < -0.1).sum()
    max_h = heights.max()
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    z_range = xyz[:, 2].max() - xyz[:, 2].min()
    print(f'  Rally {rally[\"rally_idx\"]}: h_max={max_h:.2f}m, h<-0.1={neg_h} frames, '
          f'x_span={x_range:.1f}m, z_span={z_range:.1f}m, '
          f'trimmed={rally[\"trimmed_to_ground\"]}')
"
echo ""

# ---- Copy viewer data ----
echo "=============================================="
echo "Copying viewer data..."
echo "=============================================="
cp "$OUT_DIR/rally_segmented_viewer.json" \
   "../viewer/tennis_visualizer-main/rally_data_trimmed.js"
echo "Viewer data saved to: viewer/tennis_visualizer-main/rally_data_trimmed.js"
echo "View at: tennis_synthetic.html?input=rally_data_trimmed.js"
echo ""

echo "=============================================="
echo "ALL EVALUATIONS COMPLETE at $(date)"
echo "=============================================="
