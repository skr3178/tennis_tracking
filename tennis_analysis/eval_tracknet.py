#!/usr/bin/env python3
"""Evaluate the tennis_analysis court keypoint model on the TrackNet dataset.

Mirrors the output layout of LSTM_Model/inference_output/tracknet_eval/
so results can be compared visually against TennisCourtDetector.

For each game (game1..game10), loads the first frame of Clip1, runs the
ResNet50 regression model on it, and writes a gameN_calibration.jpg with
the 14 detected keypoints drawn as red dots with green index labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from court_line_detector.court_line_detector import CourtLineDetector


def find_first_frame(clip_dir: str) -> str | None:
    if not os.path.isdir(clip_dir):
        return None
    jpgs = sorted(f for f in os.listdir(clip_dir) if f.endswith(".jpg"))
    return os.path.join(clip_dir, jpgs[0]) if jpgs else None


def draw_calibration(image: np.ndarray, keypoints: np.ndarray, title: str) -> np.ndarray:
    vis = image.copy()
    n = len(keypoints) // 2
    for i in range(n):
        x = int(round(keypoints[2 * i]))
        y = int(round(keypoints[2 * i + 1]))
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (x + 6, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(vis, title, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "models/keypoints_model.pth"))
    ap.add_argument("--dataset", default="/media/skr/storage/ten_bad/TrackNet/datasets/trackNet/Dataset")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "inference_output/tracknet_eval"))
    args = ap.parse_args()

    vis_dir = os.path.join(args.out_dir, "calibration_vis")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    detector = CourtLineDetector(args.ckpt)

    games = sorted(
        (d for d in os.listdir(args.dataset) if d.startswith("game")),
        key=lambda s: int(s.replace("game", "")),
    )
    print(f"Found {len(games)} games in {args.dataset}")

    results = {"model": "tennis_analysis/keypoints_model.pth", "games": []}
    total_ms = 0.0

    for game in games:
        clip1 = os.path.join(args.dataset, game, "Clip1")
        frame_path = find_first_frame(clip1)
        if frame_path is None:
            print(f"  {game}: no frame found, skipping")
            continue
        image = cv2.imread(frame_path)
        if image is None:
            print(f"  {game}: failed to read {frame_path}, skipping")
            continue

        t0 = time.time()
        kpts = detector.predict(image)
        dt = (time.time() - t0) * 1000.0
        total_ms += dt

        n = len(kpts) // 2
        title = f"{game}: {n}/14 keypoints detected"
        vis = draw_calibration(image, kpts, title)
        out_path = os.path.join(vis_dir, f"{game}_calibration.jpg")
        cv2.imwrite(out_path, vis)

        results["games"].append({
            "game": game,
            "frame": os.path.relpath(frame_path, args.dataset),
            "image_wh": [int(image.shape[1]), int(image.shape[0])],
            "num_keypoints": int(n),
            "inference_ms": float(dt),
            "keypoints_xy": [[float(kpts[2 * i]), float(kpts[2 * i + 1])] for i in range(n)],
        })
        print(f"  {game}: {n}/14 keypoints, {dt:.1f}ms -> {out_path}")

    results["mean_inference_ms"] = total_ms / max(1, len(results["games"]))
    json_path = os.path.join(args.out_dir, "tracknet_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {json_path}")
    print(f"Visualizations in {vis_dir}")


if __name__ == "__main__":
    main()
