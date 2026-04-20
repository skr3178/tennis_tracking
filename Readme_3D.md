# Where Is The Ball? — 3D Tennis Ball Tracking

Implementation of "Where Is The Ball?" (2025), a pipeline that lifts 2D ball detections to 3D trajectories using an LSTM-based architecture with plane-point parameterization.

> **Note:** All paths in config files (e.g. `LSTM_Model/config.py`, `parallel_collect.sh`) reference `/media/skr/storage/ten_bad/`. Update these to your local clone path before running.

## Architecture

The model (`LSTM_Model/pipeline.py`) chains four sub-networks (paper Fig. 2):

1. **EoT Network** — predicts end-of-trajectory (stroke boundary) flags from velocity deltas
2. **Height Network** — forward/backward directional LSTMs with ramp-sum mixing to estimate ball height
3. **Lift-to-3D** — closed-form ray-plane intersection: given height `h` and plane-points `P`, compute `(x, y, z)`
4. **Refinement Network** — bidirectional LSTM that corrects the initial 3D estimate

Total parameters: ~902K (~3.4 MB).

## Pipeline Overview

```
Unity Simulation → Raw Episodes (frames.csv, camera.json)
        ↓
  ours_to_npz.py  (trim to ground contacts, filter short episodes)
        ↓
  NPZ Dataset  (uv, xyz, eot, intrinsics, extrinsic per sequence)
        ↓
  train.py  (LSTM training on plane-point parameterization)
        ↓
  eval.py / eval_tracknet.py  (synthetic + real evaluation)
        ↓
  infer_video.py / infer_video_segmented.py  (real video inference)
        ↓
  viewer_3d/  (Three.js 3D visualization)
```

## Step 1: Generate Synthetic Data (Unity)

5,000 episodes were collected from the Unity tennis simulation using 24 parallel headless workers:

```bash
bash parallel_collect.sh
```

Configuration (`parallel_collect.sh`):
- Unity Editor: `~/Unity/Hub/Editor/6000.4.3f1/Editor/Unity`
- Project: `UnityProject_game/`
- Output: `TennisDataset_game_5000/test/` (4,996 episodes)
- Each episode contains: `frames.csv` (ball positions, velocities, 2D projections), `camera.json` (intrinsics + extrinsic), `meta.json`

## Step 2: Convert to NPZ Format

Trim episodes to ground-contact boundaries and convert to the model's input format:

```bash
python ours_to_npz.py \
    --split_dir TennisDataset_game_5000/test \
    --out_dir paper_npz_rev1/ours_game_5000_trimmed
```

This produces 2,417 trimmed sequences (from 4,996 raw episodes). Each `.npz` contains:
- `uv` (N, 2) — 2D ball pixel coordinates
- `xyz` (N, 3) — 3D ground-truth world coordinates
- `eot` (N,) — end-of-trajectory flags
- `intrinsics` (3,) — `[fx, cx, cy]`
- `extrinsic` (4, 4) — world-to-camera transform

## Step 3: Plane-Point Parameterization

2D pixel coordinates are converted to plane-points `P = (p_g.x, p_g.z, p_v.x, p_v.y)` (paper Eqs. 1-2) via ray-plane intersections:

- `p_g = ray ∩ {y=0}` — ground-plane intersection (drop y)
- `p_v = ray ∩ {z=0}` — vertical-plane intersection (drop z)

Implementation: `LSTM_Model/data/parameterization.py`

## Step 4: Train

```bash
cd LSTM_Model
python train.py \
    --data_root ../paper_npz_rev1 \
    --split_subdir ours_game_5000_trimmed \
    --ckpt_dir checkpoints_5k_v2 \
    --epochs 1400 \
    --batch_size 256 \
    --lr 1e-3 \
    --device cuda
```

Hyperparameters (paper Section D):
- Optimizer: Adam, lr=1e-3
- Batch size: 256
- Epochs: 1400 (converged at ~608)
- Loss: `L = 10*L_eps + 1*L_3D + 10*L_B`
  - `L_eps`: weighted BCE for EoT prediction (auto-gamma from pos/neg ratio)
  - `L_3D`: MSE on 3D coordinates
  - `L_B`: below-ground penalty (ReLU on negative y)
- UV noise augmentation: sigma=1px (training only)
- Data split: 72% train / 18% val / 10% test

Training results (608 epochs):
| Metric | Epoch 1 | Epoch 608 |
|--------|---------|-----------|
| Total loss | 509.5 | 3.86 |
| L_eps | 1.37 | 0.13 |
| L_3D | 495.9 | 2.58 |
| L_B | 0.0 | 1.08e-5 |

## Step 5: Evaluate on Synthetic Test Set

```bash
cd LSTM_Model
python eval.py \
    --ckpt checkpoints_5k_v2/best.pt \
    --split_subdir ours_game_5000_trimmed \
    --split test \
    --device cuda
```

Metrics: NRMSE on distance and height (paper Section 4.1 / Table 12).

## Step 6: Camera Calibration for Real Video

For real video inference, camera parameters are estimated using court line detection + homography + PnP:

1. **Court line detection**: `TennisCourtDetector/` detects 14 court keypoints using a trained CNN
2. **Homography refinement**: detected keypoints are refined via perspective transform against a reference court template
3. **PnP solve**: `cv2.solvePnP` with known 3D court dimensions (ITF standard: 23.77m x 10.97m) estimates camera intrinsics and extrinsic
4. **Focal length search**: sweeps focal lengths 800-3500px, selects the one with lowest reprojection error

Implementation: `LSTM_Model/eval_tracknet.py:calibrate_camera_from_image()`

## Step 7: Real Video Inference

Rally-segmented inference on real broadcast video:

```bash
cd LSTM_Model
python infer_video_segmented.py \
    --wasb_csv ../wasb_ball_positions.csv \
    --camera_json inference_output/calibrated_camera.json \
    --ckpt checkpoints_5k_v2/best.pt \
    --out_dir inference_output \
    --device cuda
```

This pipeline:
1. Loads 2D ball detections (from TrackNet / WASB)
2. Segments into rallies based on ball kinematics
3. Trims each rally to ground-contact boundaries
4. Computes plane-point parameterization using calibrated camera
5. Runs LSTM inference per rally
6. Outputs 3D trajectories + reprojection metrics

## Step 8: Visualize Results

### Three.js 3D Viewer

Open `viewer_3d/index.html` in a browser to view predicted vs ground-truth 3D trajectories. The viewer loads trajectory data from `viewer_3d/data.js`.

Generate viewer data from validation predictions:

```bash
cd LSTM_Model
python vis_val_pred.py \
    --ckpt checkpoints_5k_v2/best.pt \
    --data_root ../paper_npz_rev1 \
    --split_subdir ours_game_5000_trimmed \
    --split val \
    --out_dir inference_output/val_vis
```

## Evaluation on TrackNet Dataset

Cross-dataset evaluation on real TrackNet tennis data with automatic camera calibration:

```bash
cd LSTM_Model
python eval_tracknet.py --ckpt checkpoints_5k_v2/best.pt
```

For each game in the TrackNet dataset:
1. Calibrates camera from the first frame using court line detection
2. Extracts 2D ball positions from `Label.csv`
3. Computes plane-points and runs LSTM inference
4. Reports reprojection error and physical plausibility metrics

## Project Structure

```
LSTM_Model/
  pipeline.py              # WhereIsTheBall end-to-end forward pass
  models/
    eot_network.py         # End-of-trajectory detection (Table 5)
    height_network.py      # Directional height estimation (Tables 6-7)
    refinement_network.py  # 3D coordinate refinement (Table 8)
    lstm_blocks.py         # Shared LSTM building blocks
  lift_to_3d.py            # Closed-form height → 3D lift
  data/
    dataset.py             # NPZ dataset loader with padding
    parameterization.py    # Pixel → plane-points (Eqs. 1-2)
  losses.py                # L_eps + L_3D + L_B (Eqs. 4-7)
  config.py                # TrainConfig / EvalConfig dataclasses
  train.py                 # Training loop
  eval.py                  # Synthetic evaluation (NRMSE)
  eval_tracknet.py         # Real TrackNet evaluation with camera calibration
  infer_video.py           # Single-pass video inference
  infer_video_segmented.py # Rally-segmented video inference
  vis_val_pred.py          # Validation visualization → viewer JSON

UnityProject_game/         # Unity tennis simulation source
parallel_collect.sh        # Parallel data collection (24 workers)
ours_to_npz.py             # Raw episodes → trimmed NPZ sequences

TennisCourtDetector/       # Court keypoint detection CNN
TrackNet/                  # 2D ball detection network
viewer_3d/                 # Three.js 3D trajectory viewer
```

## Datasets and Checkpoints

- **Dataset (HuggingFace)**: https://huggingface.co/datasets/sangramrout/Tennis_3D
  - `npz/` — 2,417 trimmed sequences (15 MB)
  - `raw/raw_5000.tar.gz` — 4,996 raw Unity episodes (42 MB compressed)
- **Model checkpoint (HuggingFace)**: https://huggingface.co/sangramrout/Tennis_3D
  - `checkpoints_5k_v2/best.pt` — best model (3.5 MB, epoch 608)
- **Unity project (GitHub)**: https://github.com/skr3178/tennis_3D (`UnityProject_game/`)

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (tested with NVIDIA driver 590.48, CUDA 12.4)
- Unity 6000.4.3f1 (only needed for data generation, Step 1)

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Full dependency list (`requirements.txt`):
- PyTorch 2.6.0 (CUDA 12.4)
- OpenCV 4.11.0
- NumPy 1.26.4, SciPy 1.17.1, Pandas 3.0.2
- Matplotlib 3.10.8
- Ultralytics 8.4.37 (player pose estimation)
- tqdm, PyYAML, filterpy, omegaconf, hydra-core

### 2. Download pre-trained weights

Three model weights are required. They are not included in this repo due to size.

**LSTM checkpoint** (3D ball tracking model):
```bash
mkdir -p LSTM_Model/checkpoints_5k_v2
wget -O LSTM_Model/checkpoints_5k_v2/best.pt \
  https://huggingface.co/sangramrout/Tennis_3D/resolve/main/checkpoints_5k_v2/best.pt
```

**Court detection model** (camera calibration via court keypoints — [source repo](https://github.com/yastrebksv/TennisCourtDetector)):
```bash
# Download from Google Drive:
# https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view?usp=drive_link
# Save to: TennisCourtDetector/model_best.pt
```

**TrackNet model** (2D ball detection — [source repo](https://github.com/yastrebksv/TrackNet)):
```bash
mkdir -p TrackNet/checkpoint
# Download from Google Drive:
# https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view?usp=sharing
# Save to: TrackNet/checkpoint/model_best.pt
```

### 3. Download dataset (optional, for training)

```bash
# NPZ sequences (15 MB) — ready for training
mkdir -p paper_npz_rev1/ours_game_5000_trimmed
huggingface-cli download sangramrout/Tennis_3D \
  --repo-type dataset --local-dir paper_npz_rev1/ours_game_5000_trimmed \
  --include "npz/*"

# Raw episodes (42 MB compressed) — for re-processing
huggingface-cli download sangramrout/Tennis_3D \
  --repo-type dataset --local-dir . \
  --include "raw/*"
tar xzf raw/raw_5000.tar.gz -C TennisDataset_game_5000/test/
```

### 4. Download TrackNet dataset (optional, for real-video evaluation)

Required for `eval_tracknet.py` cross-dataset evaluation.

Download from Google Drive: https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut

See `TrackNet/README.md` for dataset structure details.
