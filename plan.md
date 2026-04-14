# Plan: Tennis Motion Capture Video Generator

## Context

Recreate the exact output of `Motion_capture.mp4` — a schematic perspective-view tennis court with pose-estimated stick-figure players, ball tracking with trajectory trails, and player movement trails. The input is `Tennis_original.mp4` (1280x720, 50fps, 16.56s broadcast footage). An old codebase at `/Users/skr3178/Downloads/ten_bad/tennis-tracking/` provides court detection, ball tracking (TrackNet), and SORT tracking that we will reuse.

---

## Tools & Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| **YOLO11n-Pose** (`ultralytics`) | latest | Player detection + 17-keypoint skeleton |
| **YOLO11n-Seg** (`ultralytics`) | latest | Player + racket instance segmentation — class 0 (person) and class 38 (tennis racket) from COCO |
| **TrackNet** (PyTorch) | pretrained checkpoint | Ball detection via heatmap CNN — 3-frame temporal input (9ch), gap-aware interpolation |
| **OpenCV** (`opencv-python`) | >=4.8.0 | Court detection (Hough lines), homography, video I/O, all drawing |
| **PyTorch** (`torch`) | >=2.0 | Unified backend for YOLO26 + TrackNet |
| **NumPy** | >=1.24,<2.0 | Array math |
| **SciPy** | >=1.10 | Savitzky-Golay smoothing for position data |
| **FilterPy** | >=1.4.5 | Kalman filter (used by SORT tracker) |
| **SymPy** | >=1.12 | Line intersection math (used by court_detector) |

---

## Files to Reuse (unchanged from existing codebase)

| File | Role |
|------|------|
| `court_detector.py` | Court line detection + homography computation per frame |
| `court_reference.py` | Court coordinate system (1665x3506 ref, baselines, service lines, net @ y=1748) |
| `sort.py` | SORT multi-object tracker (Kalman + Hungarian) |
| `utils.py` | `get_video_properties()`, `get_dtype()`, keypoint connections |

**From `/Users/skr3178/Downloads/ten_bad/TrackNet/`:**

| File | Role |
|------|------|
| `model.py` | PyTorch TrackNet model (`BallTrackerNet`) — encoder-decoder, 9ch input (3 consecutive frames) |
| `general.py` | `postprocess()` for ball detection (argmax → threshold → HoughCircles) + interpolation |
| `checkpoint/model_best.pt` | Pretrained TrackNet weights |

---

## New Files to Create

### 1. `requirements_new.txt`
Updated dependency list (drops Flask, gradio, sktime, imutils, scikit-image).

### 2. `pose_detector.py` — YOLO11-Pose Wrapper
- Loads `yolo11n-pose.pt` (auto-downloads)
- `detect(frame, conf=0.3)` → list of `(bbox, keypoints_17x3, score)`
- Replaces all Faster R-CNN code from old `detection.py`

### 2b. `racket_detector.py` — Racket Detection + Segmentation
- Loads `yolo11n-seg.pt` (auto-downloads)
- Filters detections for COCO class 38 (tennis racket)
- `detect(frame, conf=0.25)` → list of dicts with:
  - `bbox`: [x1, y1, x2, y2]
  - `mask`: binary segmentation mask (H x W)
  - `score`: confidence
- `assign_rackets_to_players(rackets, players)` — assigns each racket to nearest player based on wrist keypoint proximity
- On the schematic view: draw racket as a short line extending from the player's dominant wrist keypoint, oriented by the racket bbox angle

### 3. `schematic_renderer.py` — Perspective Court Renderer
- Computes fixed `H_ref_to_schematic` homography mapping court-reference coords → schematic output coords
- Pre-computes all court line positions in schematic space
- `draw_court(canvas)` — background, court surface trapezoid, court lines, orange net band
- `transform_keypoints(kps, game_warp_matrix)` — camera pixels → schematic coords (chain: game_warp → H_ref_to_schematic)
- `draw_skeleton(canvas, kps, color)` — draw 17-keypoint stick figure
- `draw_racket(canvas, wrist_pos, racket_angle, color)` — draw racket as short line from wrist
- `draw_trails(canvas, history, color)` — polyline from recent positions
- `render_frame(...)` — full frame composition (court + players + rackets + ball + trails + frame counter)

### 4. `generate_motion_capture.py` — Main Pipeline
Three-phase execution:

**Phase A — Detection (all 828 frames):**
1. Frame 1: `court_detector.detect(frame)` → initial homography
2. Frames 2+: `court_detector.track_court(frame)` → updated homography
3. Every frame: YOLO11-Pose → player bboxes + 17 keypoints each
3b. Every frame: YOLO11-Seg → racket bboxes + segmentation masks (COCO class 38)
3c. Assign rackets to players via wrist-keypoint proximity
4. Classify near/far player using court mask + y-position
5. Near player: proximity-based tracking (largest bbox in bottom court)
6. Far player: SORT tracker → select by max cumulative movement
7. Every 3-frame window: PyTorch TrackNet (9ch input) → ball (x,y) via heatmap + HoughCircles
   - Weights: `/Users/skr3178/Downloads/ten_bad/TrackNet/checkpoint/model_best.pt`
   - Post-processing: `postprocess()` from `TrackNet/general.py` + gap-aware interpolation

**Phase B — Post-processing:**
1. Smooth player feet positions (Savitzky-Golay, window=7, poly=2)
2. Ball outlier removal (dx>50 AND dy>50 = outlier) + linear interpolation
3. Transform all coords to court-reference space via `game_warp_matrix[i]`

**Phase C — Rendering (all frames):**
1. For each frame: `schematic_renderer.render_frame(...)` → 1280x720 BGR
2. Write to `Motion_capture_output.mp4` at 50fps via `cv2.VideoWriter`

---

## Schematic Court Geometry

**Coordinate mapping** (court_reference → schematic output):
```
Reference corners:          Schematic corners (measured from Motion_capture.mp4):
(286, 561)  far-left    →  (~406, 200)
(1379, 561) far-right   →  (~874, 200)
(286, 2935) near-left   →  (~165, 563)
(1379, 2935) near-right →  (~1114, 563)
```
Computed once via `cv2.findHomography(ref_pts, schematic_pts)`.

**Visual style:**
- Background: RGB(231, 242, 243)
- Court surface: RGB(181, 198, 207) filled trapezoid
- Court lines: RGB(100, 115, 125), 2px
- Net band: RGB(205, 140, 83), semi-transparent (alpha blend 0.6)
- Near player: green skeleton + green bbox + green circle (radius 12)
- Far player: beige skeleton + beige bbox + yellow circle (radius 8)
- Near player racket: green line (length ~25px) extending from wrist keypoint
- Far player racket: beige line (length ~18px) extending from wrist keypoint
- Ball: yellow circle (radius 5) + dark trail line (last 15 frames)
- Player trails: dark grey polyline (last 30 positions)
- Frame counter: bottom-left, small font

---

## Skeleton Keypoint Projection

All 17 YOLO26 keypoints are projected through the ground-plane homography:
```
camera_coords --[game_warp_matrix]--> court_ref_coords --[H_ref_to_schematic]--> schematic_coords
```
Combined: `H_combined = H_ref_to_schematic @ game_warp_matrix[i]`

Upper-body keypoints will have slight distortion since the homography is ground-plane only — this is acceptable and matches the reference video's appearance.

Only draw keypoints with confidence > 0.3.

---

## Verification

1. Run `python generate_motion_capture.py --input Tennis_original.mp4 --output Motion_capture_output.mp4`
2. Compare output frames side-by-side with `Motion_capture.mp4` at timestamps 0s, 4s, 8s, 12s, 16s
3. Check: court shape matches, players are in correct positions, skeletons visible, ball trail renders, frame counter increments
4. Verify output is 1280x720 @ 50fps, ~16.5s duration


# Sub Module test

Yes, absolutely. Test each component in isolation first, then integrate. Here's the order:

### Test 1 — Court Detection
- Run `court_detector.py` on `Tennis_original.mp4` frame 1
- Verify it finds the court lines and produces a valid homography matrix
- Draw detected lines on the frame to visually confirm

### Test 2 — YOLO26-Pose (Player Detection)
- Run on a single frame from `Tennis_original.mp4`
- Verify it detects both players with bounding boxes + 17 keypoints each
- Draw skeletons on the frame to confirm

### Test 2b — Racket Detection + Segmentation
- Run `yolo11n-seg` on a single frame from `Tennis_original.mp4`
- Filter for COCO class 38 (tennis racket)
- Verify it detects rackets with segmentation masks
- Draw masks + bboxes on the frame, assign to nearest player wrist

### Test 3 — PyTorch TrackNet (Ball Detection)
- Run on a few consecutive frames from `Tennis_original.mp4`
- Verify it detects the ball position
- Draw ball markers on frames to confirm

### Test 4 — Homography Transform
- Take detected player/ball positions from Tests 2-3
- Transform through the homography from Test 1
- Verify the mapped coordinates land at sensible positions on the court reference

### Test 5 — Schematic Renderer
- Draw the perspective court (static, no tracking data)
- Compare court shape against `Motion_capture.mp4`

### Then — Full Pipeline Integration

Each test is a small standalone script that saves an annotated image. If any component fails, we fix it before moving on. This avoids debugging a 300-line pipeline where the bug could be anywhere.

Want me to start with Test 1 (court detection)?