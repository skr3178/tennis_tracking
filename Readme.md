# Tennis Match Analysis Pipeline

End-to-end pipeline for tennis match analysis: court detection, player pose estimation, ball tracking, and schematic court visualization with projected player skeletons and ball trajectory.

![Side-by-side comparison](side_by_side_comparison.gif)

---

## Setup

### 1. Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Install PyTorch with CUDA 12.4
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

Key packages: `ultralytics` (YOLO pose), `opencv-python`, `scipy`, `omegaconf` (WASB config), `filterpy` (tracking). See `requirements.txt` for full list with pinned versions.

### 3. Model checkpoints

Download these checkpoints and place them in the project root:

| Model | File | Size | Download | Purpose |
|-------|------|------|----------|---------|
| YOLO26x-Pose | `yolo26x-pose.pt` | 121MB | Auto-downloaded by `ultralytics` on first run, or from [Ultralytics Assets](https://github.com/ultralytics/assets/releases) | Full-frame near player pose detection |
| YOLO11n-Pose | `yolo11n-pose.pt` | 6MB | Auto-downloaded by `ultralytics` on first run, or from [Ultralytics Assets](https://github.com/ultralytics/assets/releases) | Crop-based far player detection |
| WASB HRNet | `wasb_tennis_best.pth.tar` | 6MB | [WASB-SBDT Model Zoo](https://drive.google.com/file/d/14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z/view) (Tennis column) | Ball detection (F1=81.3) |

### 4. Input video

Default input: `S_Original_HL_clip_cropped.mp4` (1280x720, 50fps, 767 frames, crowd scene trimmed from original)

---

## Pipeline Overview

```
S_Original_HL_clip_cropped.mp4
        │
        ├─► Court Detection (frame 0, fixed camera)
        │       tennis-tracking/court_detector.py
        │       → court_warp_matrix (homography)
        │
        ├─► Player Pose Estimation (all frames)
        │       pose_detector.py (YOLO26x + YOLO11n dual-pass)
        │       → keypoints, bounding boxes, foot anchors
        │
        ├─► Ball Detection
        │       wasb_ball_detect.py (WASB HRNet)
        │       → wasb_ball_positions.csv
        │
        └─► Schematic Renderer
                generate_schematic_video.py
                → schematic_output.mp4
                  (court + player skeletons + ball trajectory)
```

---

## Running the Pipeline

### Step 1: Ball Detection (WASB)

```bash
source .venv/bin/activate

python3 wasb_ball_detect.py \
    --video S_Original_HL_clip_cropped.mp4 \
    --model wasb_tennis_best.pth.tar \
    --output wasb_ball_output.mp4 \
    --csv wasb_ball_positions.csv \
    --smooth-window 9
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | `S_Original_HL_clip_cropped.mp4` | Input video |
| `--model` | `wasb_tennis_best.pth.tar` | WASB checkpoint |
| `--output` | `wasb_ball_output.mp4` | Annotated output video |
| `--csv` | `wasb_ball_positions.csv` | Ball positions per frame |
| `--score-threshold` | `0.5` | Heatmap detection threshold |
| `--step` | `3` | Frame step (1=sliding, 3=non-overlapping) |
| `--max-interp-gap` | `15` | Max frames to interpolate across |
| `--smooth-window` | `5` | Savitzky-Golay window (odd, 0=off) |

**Pipeline:** 3-frame batched HRNet inference → heatmap blob detection → online tracker → outlier removal → linear interpolation (velocity-gated) → Savitzky-Golay smoothing (split at bounce/hit reversals)

**Result:** 97.4% frames with ball position (586 detected + 161 interpolated on 767 frames)

### Step 2: Pose Estimation Video (camera view)

```bash
python3 generate_pose_video.py
```

**Output:** `pose_estimation_output.mp4` — original video with player bounding boxes, 17-keypoint skeletons, and foot anchor dots (bbox bottom-center).

### Step 3: Schematic Court Video (players + ball)

```bash
python3 generate_schematic_video.py
```

**Output:** `schematic_output.mp4` — perspective court schematic with:
- Upright player skeletons projected via court homography
- Foot anchor markers (green=near, gold=far)
- Ball position + trailing path from `wasb_ball_positions.csv`
- Savitzky-Golay smoothing on all positions (window=7, poly=2)

**Two-pass pipeline:**
1. **Pass 1** — detect all players, collect camera-space keypoints and foot positions
2. **Pass 2** — smooth foot positions (camera-space + schematic-space), project ball, render

### Step 4: Side-by-Side Comparison

```bash
ffmpeg -y \
    -i S_Original_HL_clip_cropped.mp4 \
    -i schematic_output.mp4 \
    -filter_complex "[0:v][1:v]hstack=inputs=2,scale=1920:540" \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart \
    -an side_by_side_comparison.mp4
```

**Create GIF:**
```bash
ffmpeg -y -i side_by_side_comparison.mp4 \
    -vf "fps=15,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
    side_by_side_comparison.gif
```

---

## TrackNetV4 (Submodule)

Separate ball detection model from [TrackNetV4/TrackNetV4](https://github.com/TrackNetV4/TrackNetV4).

### Running TrackNetV2 prediction (the only compatible checkpoint)

```bash
cd TrackNetV4
source /media/skr/storage/ten_bad/.venv/bin/activate

python src/predict.py \
    --video_path /media/skr/storage/ten_bad/S_Original_HL_clip_cropped.mp4 \
    --model_weights best_model_V2_NF_RIO_1m_e8.keras \
    --output_dir output
```

**Note:** `predict.py` requires a fix — the original code references undefined classes. Replace the `custom_objects` dict in `predict.py`:

```python
model = load_model(
    model_weights_path,
    custom_objects={
        'MotionPromptLayer': MotionPromptLayer,
        'FusionLayerTypeA': FusionLayerTypeA,
        'FusionLayerTypeB': FusionLayerTypeB,
        'custom_loss': custom_loss,
    }
)
```

---

## Pipeline Components Detail

### Court Detection

Uses `tennis-tracking/court_detector.py` from the [tennis-tracking](https://github.com/yastrebksv/TennisProject) project (vendored in `tennis-tracking/` subdirectory).

**How it works:**
1. **White line detection**: grayscale threshold at 200 → Hough line transform (`minLineLength=100`, `maxLineGap=20`)
2. **Line classification**: separates horizontal/vertical lines by slope
3. **Homography estimation**: tests all combinations of 2 horizontal + 2 vertical lines against 12 reference court configurations, scores each by overlap with detected lines
4. **Output**: `court_warp_matrix` (reference court → camera) and `game_warp_matrix` (camera → reference court)

**Reference court coordinate system** (`tennis-tracking/court_reference.py`):
- Canvas: 1665x3506 pixels (court area: 1117x2408 with 274px/549px borders)
- Far baseline y=561, near baseline y=2935, net y=1748
- Left sideline x=286, right sideline x=1379
- 12 pre-defined court configurations for partial-view matching

**In this pipeline**: court detection runs once on frame 0 (fixed camera). The `court_warp_matrix` is reused for all frames, eliminating per-frame homography jitter. The schematic renderer uses this to:
- Project player foot positions from camera → schematic space
- Compute video-derived scaling by projecting sidelines into camera space

### Player Pose Estimation
`pose_detector.py` wraps YOLO-Pose with:
- **Dual-pass detection**: full-frame (`yolo26x-pose.pt`) + cropped far-player (`yolo11n-pose.pt`)
- **Court-area filtering**: rejects spectators, ball boys, umpires
- **Motion continuity tracking**: `PlayerTracker` maintains near/far player identity across frames with velocity prediction
- **Far player keypoint synthesis**: missing lower-body keypoints (hips, knees, ankles) synthesized from bounding box geometry

#### Why dual-model? YOLO pose model benchmark

Tested on frame 100 of `S_Original_HL_clip_cropped.mp4` — the far player is ~65px tall:

| Method | Far Player Score | Visible Keypoints |
|--------|-----------------|-------------------|
| yolo26x full frame | **Not detected** | - |
| yolo26x cropped+upscaled | 0.16-0.26 | 13/17 |
| **yolo11n cropped+upscaled** | **0.78-0.86** | **17/17** |

The larger yolo26x model is tuned for full-resolution figures and struggles with 4x upscaled crop artifacts. The nano model (`yolo11n`) generalizes better at this scale, achieving higher confidence and full keypoint visibility on the far player. For the near player, yolo26x on the full frame works well (score=0.85, 13 visible keypoints).

### Schematic Renderer
`schematic_renderer.py` renders a perspective court view with:
- **Video-derived scaling**: `pixel_scale = schematic_court_width / camera_court_width` at each depth — no hardcoded constants. Court sidelines are projected into camera space from frame 0 and used as interpolation lookup tables.
- **Ground-plane homography**: only foot positions go through the homography; skeletons are drawn upright from the foot anchor
- **Dynamic net height**: derived from `NET_HEIGHT_REF / REF_COURT_WIDTH` ratio applied to local schematic court width

---

## Key Files

| File | Description |
|------|-------------|
| `generate_schematic_video.py` | Main pipeline: two-pass detection + smoothing + schematic rendering with ball |
| `generate_pose_video.py` | Camera-view pose estimation with foot anchors |
| `wasb_ball_detect.py` | WASB HRNet ball detection and tracking |
| `schematic_renderer.py` | Perspective court renderer with video-derived scaling |
| `pose_detector.py` | YOLO pose detection with dual-pass + court filtering |
| `ball_tracker.py` | Ball tracking with interpolation and smoothing |
| `racket_detector.py` | Racket detection/segmentation |
| `generate_motion_capture.py` | Full motion capture pipeline (detection + post-processing + rendering) |
| `test_components.py` | Visual tests for each pipeline component |
| `tennis-tracking/court_detector.py` | Court line detection and homography |
| `tennis-tracking/court_reference.py` | Reference court coordinates |

---

## Known Issues

- **TrackNetV4 checkpoint incompatibility** ([issue #6](https://github.com/TrackNetV4/TrackNetV4/issues/6)): The V4 model checkpoints were saved with old class names (`MotionIncorporationLayerV1`, `CombineOutputs`, `MotionFramesInput`) that don't exist in the current repo code. Only `best_model_V2_NF_RIO_1m_e8.keras` (plain TrackNetV2) loads cleanly.
- **TrackNetV4 dataset splits swapped** ([issue #7](https://github.com/TrackNetV4/TrackNetV4/issues/7)): `_process_clip_level()` and `_process_game_level()` use each other's CSV files.

---

## References

- [mmpose](https://github.com/open-mmlab/mmpose) — pose estimation models
- [TrackNetV4](https://github.com/TrackNetV4/TrackNetV4) — ball detection
- [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) — ball detection benchmark
- Pose model comparison: `pose_model_benchmark_comparison.html`
