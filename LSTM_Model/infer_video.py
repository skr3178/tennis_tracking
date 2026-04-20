#!/usr/bin/env python3
"""End-to-end inference: video → 2D ball detection → plane-points → LSTM → 3D trajectory.

Usage:
    python infer_video.py --video ../S_Original_HL_clip_cropped.mp4 \
                          --ckpt checkpoints_combined/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _parent)
sys.path.insert(0, os.path.join(_parent, "TrackNet"))
from ball_tracker import BallTracker

from data.parameterization import pixel_to_plane_points
from pipeline import WhereIsTheBall


def estimate_broadcast_camera(width=1280, height=720, fov_y_deg=25.0,
                               eye=(0.0, 6.0, -10.0), look=(0.0, 0.5, 8.0)):
    """Estimate a broadcast-style camera for a real tennis video.
    Returns intrinsics (3,), extrinsic (4,4), and metadata dict."""
    fy = (height * 0.5) / np.tan(np.radians(fov_y_deg) * 0.5)
    fx = fy
    cx, cy = width * 0.5, height * 0.5

    eye = np.array(eye, dtype=np.float64)
    look_pt = np.array(look, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0])

    f = (look_pt - eye)
    f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, -f], axis=0)
    t = -R @ eye
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R
    E[:3, 3] = t

    return (
        np.array([fx, cx, cy], dtype=np.float32),
        E.astype(np.float32),
        {"width": width, "height": height, "fov_y_deg": fov_y_deg,
         "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
         "eye": list(eye), "look": list(look_pt),
         "convention": "opengl"}
    )


def segment_rallies(uv: np.ndarray, max_gap: int = 15,
                    min_rally_len: int = 10) -> list[np.ndarray]:
    """Split a detection sequence into rally segments based on detection gaps."""
    valid = np.array([i for i in range(len(uv)) if not np.isnan(uv[i]).any()])
    if len(valid) < min_rally_len:
        return []

    segments = []
    seg_start = 0
    for i in range(1, len(valid)):
        if valid[i] - valid[i - 1] > max_gap:
            if i - seg_start >= min_rally_len:
                segments.append(valid[seg_start:i])
            seg_start = i
    if len(valid) - seg_start >= min_rally_len:
        segments.append(valid[seg_start:])

    return segments


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Ball detection ---
    print(f"[1/4] Detecting ball in {args.video} ...")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"    {len(frames)} frames, {width}x{height}, {fps:.1f} fps")

    tracker = BallTracker(
        weights_path=os.path.join(os.path.dirname(__file__),
                                  "..", "TrackNet", "checkpoint", "model_best.pt"),
        device=str(device),
    )
    raw_pos = tracker.detect_sequence(frames)
    det_count = sum(1 for p in raw_pos if p is not None)
    print(f"    Raw detections: {det_count}/{len(frames)}")

    positions = tracker.remove_outliers(raw_pos)
    positions = tracker.interpolate_positions(positions)
    positions = tracker.smooth_positions(positions)
    det_count2 = sum(1 for p in positions if p is not None)
    print(f"    After cleanup:  {det_count2}/{len(frames)}")

    uv_full = np.full((len(frames), 2), np.nan, dtype=np.float32)
    for i, p in enumerate(positions):
        if p is not None:
            uv_full[i] = [p[0], p[1]]

    # --- Step 2: Camera + parameterization ---
    print("[2/4] Computing plane-points parameterization ...")
    intrinsics, extrinsic, cam_meta = estimate_broadcast_camera(
        width, height, fov_y_deg=args.fov_y_deg,
        eye=tuple(args.cam_eye), look=tuple(args.cam_look),
    )
    print(f"    Camera: fov={args.fov_y_deg}°  eye={args.cam_eye}  look={args.cam_look}")

    rally_segments = segment_rallies(uv_full, max_gap=args.max_gap,
                                     min_rally_len=args.min_rally_len)
    print(f"    Found {len(rally_segments)} rally segments")

    # --- Step 3: LSTM inference ---
    print(f"[3/4] Running LSTM model ({args.ckpt}) ...")
    net = WhereIsTheBall(hidden=64).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(state["model_state"])
    net.eval()
    print(f"    Loaded epoch {state.get('epoch', '?')}")

    all_results = []
    for seg_idx, seg_frames in enumerate(rally_segments):
        uv_seg = uv_full[seg_frames]
        P = pixel_to_plane_points(uv_seg, intrinsics, extrinsic, convention="opengl")

        if np.isnan(P).any():
            nan_frac = np.isnan(P).any(axis=1).mean()
            print(f"    Segment {seg_idx}: {nan_frac:.0%} NaN plane-points — skipping")
            continue

        P_t = torch.from_numpy(P).unsqueeze(0).to(device)
        lengths = torch.tensor([len(P)], dtype=torch.long)

        with torch.no_grad():
            out = net(P_t, lengths=lengths)

        xyz = out["xyz_final"][0].cpu().numpy()
        eps = out["eps"][0].cpu().numpy().squeeze(-1)
        h = out["h_refined"][0].cpu().numpy().squeeze(-1)

        all_results.append({
            "segment_idx": seg_idx,
            "frame_indices": seg_frames.tolist(),
            "uv": uv_seg,
            "xyz": xyz,
            "eps": eps,
            "height": h,
            "P": P,
        })
        print(f"    Segment {seg_idx}: {len(seg_frames)} frames  "
              f"x=[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}]  "
              f"y=[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}]  "
              f"z=[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]")

    if not all_results:
        print("No valid rally segments found. Try adjusting --min_rally_len or camera params.")
        return

    # --- Step 4: Visualization ---
    print(f"[4/4] Generating visualizations ...")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 4a: 3D trajectory plot
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"LSTM 3D Ball Trajectory — {os.path.basename(args.video)}", fontsize=14)

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_results), 1)))

    for i, res in enumerate(all_results):
        xyz = res["xyz"]
        c = colors[i % len(colors)]
        label = f"seg {res['segment_idx']} ({len(res['frame_indices'])}f)"

        ax1.plot(xyz[:, 0], xyz[:, 2], xyz[:, 1], "-", color=c, label=label, linewidth=1.5)
        ax1.scatter(xyz[0, 0], xyz[0, 2], xyz[0, 1], color=c, s=40, marker="o")

        ax2.plot(xyz[:, 0], xyz[:, 2], "-", color=c, label=label, linewidth=1.5)
        ax2.scatter(xyz[0, 0], xyz[0, 2], color=c, s=30, marker="o")

        ax3.plot(xyz[:, 2], xyz[:, 1], "-", color=c, label=label, linewidth=1.5)
        ax3.scatter(xyz[0, 2], xyz[0, 1], color=c, s=30, marker="o")

        ax4.plot(res["frame_indices"], xyz[:, 1], "-", color=c, label=label, linewidth=1.5)

    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)"); ax1.set_zlabel("Y / height (m)")
    ax1.set_title("3D trajectory"); ax1.legend(fontsize=7)

    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)")
    ax2.set_title("Top-down (X-Z)"); ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)

    ax3.set_xlabel("Z (m)"); ax3.set_ylabel("Y / height (m)")
    ax3.set_title("Side view (Z-Y)"); ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)

    ax4.set_xlabel("Frame"); ax4.set_ylabel("Y / height (m)")
    ax4.set_title("Height over time"); ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "trajectory_3d.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    Saved {plot_path}")

    # 4b: Video with overlay
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_path = os.path.join(out_dir, "inference_overlay.mp4")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

    frame_to_xyz = {}
    frame_to_seg = {}
    for res in all_results:
        for j, fi in enumerate(res["frame_indices"]):
            frame_to_xyz[fi] = res["xyz"][j]
            frame_to_seg[fi] = res["segment_idx"]

    trail_len = 15
    for fi, frame in enumerate(frames):
        out_frame = frame.copy()

        if not np.isnan(uv_full[fi]).any():
            u, v = int(uv_full[fi, 0]), int(uv_full[fi, 1])
            cv2.circle(out_frame, (u, v), 6, (0, 255, 255), 2)

        for back in range(1, trail_len):
            prev = fi - back
            if prev in frame_to_xyz and fi in frame_to_xyz:
                if not np.isnan(uv_full[prev]).any() and not np.isnan(uv_full[fi - back + 1]).any():
                    p1 = (int(uv_full[prev, 0]), int(uv_full[prev, 1]))
                    p2 = (int(uv_full[fi - back + 1, 0]), int(uv_full[fi - back + 1, 1]))
                    alpha = 1.0 - back / trail_len
                    color = (0, int(200 * alpha), int(255 * alpha))
                    cv2.line(out_frame, p1, p2, color, 2)

        if fi in frame_to_xyz:
            xyz = frame_to_xyz[fi]
            info = f"X={xyz[0]:.2f}  Y={xyz[1]:.2f}  Z={xyz[2]:.2f}"
            cv2.putText(out_frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(out_frame, f"Frame {fi}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(out_frame)

    writer.release()
    print(f"    Saved {overlay_path}")

    # 4c: Save raw data as JSON
    json_path = os.path.join(out_dir, "trajectory_3d.json")
    export = {
        "video": args.video,
        "fps": fps,
        "num_frames": len(frames),
        "camera": cam_meta,
        "model_ckpt": args.ckpt,
        "model_epoch": state.get("epoch"),
        "segments": [],
    }
    for res in all_results:
        export["segments"].append({
            "segment_idx": res["segment_idx"],
            "frame_indices": res["frame_indices"],
            "uv": res["uv"].tolist(),
            "xyz": res["xyz"].tolist(),
            "eps": res["eps"].tolist(),
            "height": res["height"].tolist(),
        })
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"    Saved {json_path}")

    print(f"\nDone! {len(all_results)} segments, "
          f"{sum(len(r['frame_indices']) for r in all_results)} total frames with 3D output.")


def main():
    p = argparse.ArgumentParser(description="Video → 3D ball trajectory inference")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--ckpt", default="checkpoints_combined/best.pt")
    p.add_argument("--out_dir", default="inference_output")
    p.add_argument("--fov_y_deg", type=float, default=25.0)
    p.add_argument("--cam_eye", type=float, nargs=3, default=[0.0, 6.0, -10.0])
    p.add_argument("--cam_look", type=float, nargs=3, default=[0.0, 0.5, 8.0])
    p.add_argument("--max_gap", type=int, default=15,
                   help="Max frame gap before splitting into new rally segment")
    p.add_argument("--min_rally_len", type=int, default=10,
                   help="Min frames to consider a valid rally segment")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
