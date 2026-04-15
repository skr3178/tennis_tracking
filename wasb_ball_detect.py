#!/usr/bin/env python3
"""
Standalone WASB tennis ball detection and tracking on arbitrary video.

Uses the WASB (HRNet) model from the WASB-SBDT repo to detect and track
a tennis ball frame-by-frame. Outputs a CSV of ball coordinates and an
annotated video.

Usage:
    python wasb_ball_detect.py \
        --video S_Original_HL_clip_cropped.mp4 \
        --model wasb_tennis_best.pth.tar \
        --output wasb_ball_output.mp4
"""

import sys
import os
import argparse
import csv
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

# Add WASB-SBDT src to path so we can import its model module
WASB_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WASB-SBDT", "src")
sys.path.insert(0, WASB_SRC)

from models.hrnet import HRNet

# Import affine helpers directly from the file to avoid heavy transitive imports
import importlib.util
_spec = importlib.util.spec_from_file_location("wasb_image", os.path.join(WASB_SRC, "utils", "image.py"))
_image_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_image_mod)
get_affine_transform = _image_mod.get_affine_transform
affine_transform = _image_mod.affine_transform

# ---------------------------------------------------------------------------
# WASB model config (from configs/model/wasb.yaml)
# ---------------------------------------------------------------------------
WASB_MODEL_CFG = {
    "name": "hrnet",
    "frames_in": 3,
    "frames_out": 3,
    "inp_height": 288,
    "inp_width": 512,
    "out_height": 288,
    "out_width": 512,
    "rgb_diff": False,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": "SUM",
            },
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}

# ImageNet normalization
NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

INPUT_W, INPUT_H = 512, 288
SCORE_THRESHOLD = 0.5
MAX_TRACKER_DISP = 300


# ---------------------------------------------------------------------------
# Postprocessor: sigmoid + connected-components blob detection
# ---------------------------------------------------------------------------
def detect_blobs(heatmap, score_threshold=SCORE_THRESHOLD):
    """Detect ball candidates from a single-frame heatmap via connected components."""
    xys, scores = [], []
    if np.max(heatmap) > score_threshold:
        _, hm_th = cv2.threshold(heatmap, score_threshold, 1, cv2.THRESH_BINARY)
        n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
        for m in range(1, n_labels):
            ys, xs = np.where(labels == m)
            ws = heatmap[ys, xs]
            score = ws.sum()
            x = np.sum(xs.astype(np.float64) * ws) / np.sum(ws)
            y = np.sum(ys.astype(np.float64) * ws) / np.sum(ws)
            xys.append(np.array([x, y], dtype=np.float64))
            scores.append(float(score))
    return xys, scores


# ---------------------------------------------------------------------------
# Simple online tracker (reimplemented from trackers/online.py)
# ---------------------------------------------------------------------------
class SimpleTracker:
    def __init__(self, max_disp=MAX_TRACKER_DISP):
        self.max_disp = max_disp
        self.fid = 0
        self.history = {}  # fid -> {xy, visi, score}

    def update(self, detections):
        """detections: list of {'xy': np.array([x,y]), 'score': float}"""
        # Filter by max displacement from previous position
        if self.fid > 0 and self.history.get(self.fid - 1, {}).get("visi", False):
            prev_xy = self.history[self.fid - 1]["xy"]
            detections = [d for d in detections
                          if np.linalg.norm(d["xy"] - prev_xy) < self.max_disp]

        # Predict next position using quadratic motion model
        xy_pred = self._predict()

        # Select best detection
        best_score = -np.inf
        best_xy = np.array([-np.inf, -np.inf])
        visi = False

        for det in detections:
            score = det["score"]
            if xy_pred is not None:
                score += -np.linalg.norm(xy_pred - det["xy"])
            if score > best_score:
                best_score = score
                best_xy = det["xy"]
                visi = True

        self.history[self.fid] = {"xy": best_xy, "visi": visi, "score": best_score}
        self.fid += 1
        return {"x": float(best_xy[0]), "y": float(best_xy[1]),
                "visi": visi, "score": float(best_score)}

    def _predict(self):
        f = self.fid
        h = self.history
        if (f >= 3
                and h.get(f-1, {}).get("visi", False)
                and h.get(f-2, {}).get("visi", False)
                and h.get(f-3, {}).get("visi", False)):
            xy1 = h[f-1]["xy"]
            xy2 = h[f-2]["xy"]
            xy3 = h[f-3]["xy"]
            acc = (xy1 - xy2) - (xy2 - xy3)
            vel = (xy1 - xy2) + acc
            return xy1 + vel + acc / 2
        return None


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def compute_affine(frame_bgr):
    """Compute forward and inverse affine transforms for a frame."""
    h, w = frame_bgr.shape[:2]
    c = np.array([w / 2., h / 2.], dtype=np.float32)
    s = max(h, w) * 1.0
    trans_fwd = get_affine_transform(c, s, 0, [INPUT_W, INPUT_H], inv=0)
    trans_inv = get_affine_transform(c, s, 0, [INPUT_W, INPUT_H], inv=1)
    return trans_fwd, trans_inv


def preprocess_frame(frame_bgr, trans_fwd):
    """Warp frame to model input size and normalize."""
    warped = cv2.warpAffine(frame_bgr, trans_fwd, (INPUT_W, INPUT_H),
                            flags=cv2.INTER_LINEAR)
    # BGR -> RGB -> PIL -> normalize
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = NORMALIZE(pil_img)  # (3, 288, 512)
    return tensor


# ---------------------------------------------------------------------------
# Interpolation and smoothing
# ---------------------------------------------------------------------------
def interpolate_and_smooth(results, n_frames, max_gap=15, smooth_window=5):
    """
    1. Remove outlier detections that jump too far from neighbors.
    2. Fill short gaps using linear interpolation (no overshoot).
    3. Apply Savitzky-Golay filter to smooth the full trajectory.
    """
    from scipy.signal import savgol_filter

    # --- Step 0: Collect detected positions ---
    det_frames = []
    det_x = []
    det_y = []
    for i in range(n_frames):
        r = results[i]
        if r["visi"]:
            det_frames.append(i)
            det_x.append(r["x"])
            det_y.append(r["y"])

    if len(det_frames) < 4:
        return results

    det_frames = np.array(det_frames)
    det_x = np.array(det_x)
    det_y = np.array(det_y)

    # --- Step 1: Remove outlier jumps ---
    # If a detection jumps far from both its predecessor and successor,
    # it's likely a false positive (e.g., crowd, scoreboard).
    max_px_per_frame = 40.0  # max plausible ball speed in px/frame
    outliers_removed = 0
    for idx in range(1, len(det_frames) - 1):
        dt_prev = det_frames[idx] - det_frames[idx - 1]
        dt_next = det_frames[idx + 1] - det_frames[idx]
        dist_prev = np.hypot(det_x[idx] - det_x[idx-1], det_y[idx] - det_y[idx-1])
        dist_next = np.hypot(det_x[idx+1] - det_x[idx], det_y[idx+1] - det_y[idx])
        speed_prev = dist_prev / max(dt_prev, 1)
        speed_next = dist_next / max(dt_next, 1)
        # Both neighbors far away → outlier
        if speed_prev > max_px_per_frame and speed_next > max_px_per_frame:
            results[det_frames[idx]]["visi"] = False
            outliers_removed += 1

    if outliers_removed:
        print(f"  Removed {outliers_removed} outlier detections")

    # Rebuild clean detection list
    det_frames = []
    det_x = []
    det_y = []
    for i in range(n_frames):
        r = results[i]
        if r["visi"]:
            det_frames.append(i)
            det_x.append(r["x"])
            det_y.append(r["y"])

    det_frames = np.array(det_frames)
    det_x = np.array(det_x)
    det_y = np.array(det_y)

    # --- Step 2: Linear interpolation for short gaps ---
    # Only interpolate if the velocity across the gap is reasonable.
    interp_count = 0
    for idx in range(len(det_frames) - 1):
        f_start = int(det_frames[idx])
        f_end = int(det_frames[idx + 1])
        gap_len = f_end - f_start - 1
        if gap_len <= 0 or gap_len > max_gap:
            continue

        # Check that the velocity across the gap is plausible
        dist = np.hypot(det_x[idx+1] - det_x[idx], det_y[idx+1] - det_y[idx])
        speed = dist / (f_end - f_start)
        if speed > max_px_per_frame:
            continue  # skip — likely a scene change or serve bounce

        # Linear interpolation
        for f in range(f_start + 1, f_end):
            t = (f - f_start) / (f_end - f_start)
            ix = det_x[idx] + t * (det_x[idx+1] - det_x[idx])
            iy = det_y[idx] + t * (det_y[idx+1] - det_y[idx])
            results[f] = {
                "x": float(ix),
                "y": float(iy),
                "visi": True,
                "score": 0.0,
                "interpolated": True,
            }
            interp_count += 1

    print(f"  Interpolated {interp_count} gap frames (max_gap={max_gap})")

    # --- Step 3: Savitzky-Golay smoothing on contiguous runs ---
    # Split runs at direction reversals (bounces / hits) so the smoother
    # doesn't blur across sharp physical transitions.
    if smooth_window >= 3:
        vis_frames = []
        vis_x = []
        vis_y = []
        for i in range(n_frames):
            r = results[i]
            if r["visi"]:
                vis_frames.append(i)
                vis_x.append(r["x"])
                vis_y.append(r["y"])

        vis_x = np.array(vis_x)
        vis_y = np.array(vis_y)

        # Find contiguous runs (split at temporal gaps)
        runs = []
        run_start = 0
        for j in range(1, len(vis_frames)):
            if vis_frames[j] != vis_frames[j - 1] + 1:
                runs.append((run_start, j))
                run_start = j
        runs.append((run_start, len(vis_frames)))

        # Further split each run at direction-reversal points.
        # A reversal is where the y-velocity sign flips significantly,
        # indicating a bounce or racket hit.
        min_reversal_dy = 3.0  # px — ignore sub-pixel noise
        sub_runs = []
        for rs, re in runs:
            if re - rs < 5:
                sub_runs.append((rs, re))
                continue
            # Compute y velocity
            dy = np.diff(vis_y[rs:re])
            splits = [rs]
            for j in range(1, len(dy)):
                # sign flip with meaningful magnitude on both sides
                if (dy[j-1] * dy[j] < 0
                        and abs(dy[j-1]) > min_reversal_dy
                        and abs(dy[j]) > min_reversal_dy):
                    splits.append(rs + j)
            splits.append(re)
            for k in range(len(splits) - 1):
                sub_runs.append((splits[k], splits[k+1]))

        for rs, re in sub_runs:
            run_len = re - rs
            win = smooth_window
            if win % 2 == 0:
                win += 1
            if run_len >= win:
                poly_order = min(2, win - 1)
                vis_x[rs:re] = savgol_filter(vis_x[rs:re], win, poly_order)
                vis_y[rs:re] = savgol_filter(vis_y[rs:re], win, poly_order)

        # Write smoothed values back
        for j, f in enumerate(vis_frames):
            results[f]["x"] = float(vis_x[j])
            results[f]["y"] = float(vis_y[j])

        print(f"  Applied Savitzky-Golay smoothing (window={smooth_window}, "
              f"{len(sub_runs)} segments)")

    return results


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WASB tennis ball detection on video")
    parser.add_argument("--video", type=str, default="S_Original_HL_clip_cropped.mp4",
                        help="Input video path")
    parser.add_argument("--model", type=str, default="wasb_tennis_best.pth.tar",
                        help="Path to WASB model checkpoint (.pth.tar)")
    parser.add_argument("--output", type=str, default="wasb_ball_output.mp4",
                        help="Output annotated video path")
    parser.add_argument("--csv", type=str, default="wasb_ball_positions.csv",
                        help="Output CSV with ball positions per frame")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD,
                        help="Heatmap score threshold for blob detection")
    parser.add_argument("--step", type=int, default=3,
                        help="Detector step size (3 = process every 3rd frame set)")
    parser.add_argument("--max-interp-gap", type=int, default=15,
                        help="Max gap length (frames) to interpolate across")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Savitzky-Golay smoothing window (odd, 0=disable)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load model ---
    print(f"Loading WASB model from {args.model} ...")
    model = HRNet(OmegaConf.create(WASB_MODEL_CFG))
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0])
    model.eval()
    print("Model loaded.")

    # --- Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {orig_w}x{orig_h} @ {fps:.1f} fps, {total_frames} frames")

    # --- Video writer (H.264) ---
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        # Fallback if avc1 is not available
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))

    # --- Read all frames ---
    print("Reading frames...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Read {len(frames)} frames.")

    # Compute affine transforms (same for all frames since resolution is constant)
    trans_fwd, trans_inv = compute_affine(frames[0])

    # Preprocess all frames into tensors
    print("Preprocessing frames...")
    frame_tensors = []
    for frame in frames:
        frame_tensors.append(preprocess_frame(frame, trans_fwd))

    # --- Run inference ---
    # The model takes 3 consecutive frames as input (concatenated along channels)
    # and outputs 3 heatmaps, one per frame.
    # With step=3, we process frames [0,1,2], [3,4,5], [6,7,8], ...
    # With step=1, we process frames [0,1,2], [1,2,3], [2,3,4], ... (sliding window)
    print(f"Running inference (step={args.step})...")
    tracker = SimpleTracker()
    results = {}  # frame_idx -> {x, y, visi, score}

    n_frames = len(frames)
    frames_in = 3
    step = args.step

    # Build list of (start_idx, frame_indices) for each batch of 3 frames
    batch_groups = []
    idx = 0
    while idx + frames_in - 1 < n_frames:
        group = list(range(idx, idx + frames_in))
        batch_groups.append(group)
        idx += step

    # Handle remaining frames at the end
    if batch_groups and batch_groups[-1][-1] < n_frames - 1:
        # Add one more group covering the last frames
        last_start = n_frames - frames_in
        if last_start > batch_groups[-1][0]:
            batch_groups.append(list(range(last_start, last_start + frames_in)))

    # Process in batches for efficiency
    batch_size = 8
    all_detections = {}  # frame_idx -> list of detections

    with torch.no_grad():
        for bi in range(0, len(batch_groups), batch_size):
            batch_group = batch_groups[bi:bi + batch_size]
            imgs_batch = []
            for group in batch_group:
                cat = torch.cat([frame_tensors[i] for i in group], dim=0)  # (9, 288, 512)
                imgs_batch.append(cat)

            imgs_tensor = torch.stack(imgs_batch, dim=0).to(device)  # (B, 9, 288, 512)
            preds = model(imgs_tensor)  # {0: (B, 3, 288, 512)}

            heatmaps = preds[0].sigmoid_().cpu().numpy()  # (B, 3, 288, 512)

            for gi, group in enumerate(batch_group):
                for fi, frame_idx in enumerate(group):
                    hm = heatmaps[gi, fi]
                    xys, scores = detect_blobs(hm, args.score_threshold)

                    # Transform detected coordinates back to original image space
                    dets = []
                    for xy, score in zip(xys, scores):
                        xy_orig = affine_transform(xy.astype(np.float32), trans_inv)
                        dets.append({"xy": xy_orig, "score": score})

                    # Keep only the first detection set for each frame
                    # (later groups may re-detect the same frame with step < frames_in)
                    if frame_idx not in all_detections:
                        all_detections[frame_idx] = dets

            if (bi // batch_size) % 10 == 0:
                processed = min(bi + batch_size, len(batch_groups))
                print(f"  Processed {processed}/{len(batch_groups)} groups "
                      f"({processed * step}/{n_frames} frames approx)")

    # Run tracker sequentially over all frames
    print("Running tracker...")
    for frame_idx in range(n_frames):
        dets = all_detections.get(frame_idx, [])
        result = tracker.update(dets)
        results[frame_idx] = result

    # --- Interpolate gaps and smooth trajectory ---
    print("Smoothing trajectory...")
    results = interpolate_and_smooth(results, n_frames,
                                     max_gap=args.max_interp_gap,
                                     smooth_window=args.smooth_window)

    # --- Write output ---
    print("Writing output video and CSV...")
    with open(args.csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "x", "y", "visible", "interpolated", "score"])

        for frame_idx, frame in enumerate(frames):
            r = results[frame_idx]
            vis_frame = frame.copy()

            if r["visi"]:
                x, y = int(round(r["x"])), int(round(r["y"]))
                interp = r.get("interpolated", False)
                if interp:
                    # Yellow circle for interpolated positions
                    cv2.circle(vis_frame, (x, y), 8, (0, 255, 255), 2)
                    cv2.circle(vis_frame, (x, y), 2, (0, 255, 255), -1)
                else:
                    # Green circle for detected positions
                    cv2.circle(vis_frame, (x, y), 8, (0, 255, 0), 2)
                    cv2.circle(vis_frame, (x, y), 2, (0, 255, 0), -1)
                csv_writer.writerow([frame_idx, f"{r['x']:.1f}", f"{r['y']:.1f}",
                                     1, int(interp), f"{r['score']:.3f}"])
            else:
                csv_writer.writerow([frame_idx, "", "", 0, 0, ""])

            writer.write(vis_frame)

    writer.release()
    print(f"Done! Output video: {args.output}")
    print(f"Ball positions CSV: {args.csv}")

    # Print summary
    visible_count = sum(1 for r in results.values() if r["visi"])
    interp_count = sum(1 for r in results.values() if r.get("interpolated", False))
    detected_count = visible_count - interp_count
    print(f"Ball visible in {visible_count}/{n_frames} frames "
          f"({100*visible_count/n_frames:.1f}%) — "
          f"{detected_count} detected, {interp_count} interpolated")


if __name__ == "__main__":
    main()
