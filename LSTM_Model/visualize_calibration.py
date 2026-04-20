#!/usr/bin/env python3
"""Visualize court keypoint detection + homography correction for all TrackNet games.

For each game, draws:
  - Blue dots: raw detected keypoints
  - Green dots: homography-corrected keypoints
  - Court lines connecting the corrected keypoints
  - Title with detection count and camera parameters

Usage:
    python visualize_calibration.py
    python visualize_calibration.py --games 1,3,7
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _parent)
sys.path.insert(0, os.path.join(_parent, "TennisCourtDetector"))

from tracknet import BallTrackerNet
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix
from court_reference import CourtReference

COURT_LINES = [
    (0, 1), (2, 3), (0, 2), (1, 3),
    (4, 5), (6, 7),
    (4, 6), (8, 9), (10, 11), (5, 7),
    (12, 13),
]


def detect_and_visualize(image_path: str, model: torch.nn.Module, device: torch.device):
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    h, w = image.shape[:2]
    img = cv2.resize(image, (640, 360))
    inp = (img.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0)

    out = model(inp.float().to(device))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()

    post_scale_x = w / 1280.0
    post_scale_y = h / 720.0

    raw_kps = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if x_pred is not None and y_pred is not None:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            raw_kps.append((x_pred * post_scale_x, y_pred * post_scale_y))
        else:
            raw_kps.append((None, None))

    detected = sum(1 for p in raw_kps if p[0] is not None)
    if detected < 4:
        return None, None

    matrix = get_trans_matrix(raw_kps)
    corrected_kps = None
    if matrix is not None:
        court_ref = CourtReference()
        refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))
        corrected = cv2.perspectiveTransform(refer_kps, matrix)
        corrected_kps = [tuple(corrected[i].flatten()) for i in range(14)]

    vis = image.copy()

    if corrected_kps is not None:
        for i, j in COURT_LINES:
            pt1 = corrected_kps[i]
            pt2 = corrected_kps[j]
            if (0 <= pt1[0] <= w and 0 <= pt1[1] <= h and
                    0 <= pt2[0] <= w and 0 <= pt2[1] <= h):
                cv2.line(vis, (int(pt1[0]), int(pt1[1])),
                         (int(pt2[0]), int(pt2[1])), (0, 255, 0), 1)
        for i, pt in enumerate(corrected_kps):
            if 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

    for i, (px, py) in enumerate(raw_kps):
        if px is not None:
            cv2.circle(vis, (int(px), int(py)), 4, (255, 0, 0), -1)
            cv2.putText(vis, str(i), (int(px) + 6, int(py) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    corr_str = "homography corrected" if corrected_kps else "no homography"
    title = f"Blue: raw detected ({detected}/14), Green: {corr_str}"
    cv2.putText(vis, title, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return vis, {"detected": detected, "has_homography": corrected_kps is not None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracknet_dir", default="../TrackNet/datasets/trackNet/Dataset")
    p.add_argument("--court_model", default="../TennisCourtDetector/model_best.pt")
    p.add_argument("--out_dir", default="inference_output/tracknet_eval_homography/calibration_vis")
    p.add_argument("--device", default="cuda")
    p.add_argument("--games", type=str, default=None,
                   help="Comma-separated game numbers (e.g., '1,3,7'). Default: all.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = BallTrackerNet(out_channels=15)
    model = model.to(device)
    model.load_state_dict(torch.load(args.court_model, map_location=device))
    model.eval()

    games = sorted([d for d in os.listdir(args.tracknet_dir)
                    if d.startswith("game") and os.path.isdir(os.path.join(args.tracknet_dir, d))],
                   key=lambda s: int(s.replace("game", "")))
    if args.games:
        selected = set(f"game{g}" for g in args.games.split(","))
        games = [g for g in games if g in selected]

    print(f"Generating calibration visualizations for {len(games)} games...")

    for game in games:
        game_dir = os.path.join(args.tracknet_dir, game)
        clip1 = os.path.join(game_dir, "Clip1")
        frames = sorted(f for f in os.listdir(clip1) if f.endswith(".jpg"))
        if not frames:
            print(f"  {game}: no frames, skipping")
            continue
        frame_path = os.path.join(clip1, frames[0])

        vis, info = detect_and_visualize(frame_path, model, device)
        if vis is None:
            print(f"  {game}: detection failed")
            continue

        out_path = os.path.join(args.out_dir, f"{game}_calibration.png")
        cv2.imwrite(out_path, vis)
        print(f"  {game}: {info['detected']}/14 keypoints, homography={'yes' if info['has_homography'] else 'no'} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
