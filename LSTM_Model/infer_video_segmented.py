#!/usr/bin/env python3
"""Rally-segmented inference: WASB CSV → rally segmentation → plane-points → LSTM → 3D + viewer.

Implements proper rally-level segmentation per the paper's approach:
- Each input to the LSTM is a full rally (serve-to-dead-ball, multiple strokes)
- First/last frames trimmed to ground contact (bounce points)
- EoT network handles stroke boundaries within each rally
"""
from __future__ import annotations

import argparse
import json
import csv
import sys
import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LSTM_Model.data.parameterization import pixel_to_plane_points
from LSTM_Model.pipeline import WhereIsTheBall


def load_wasb_detections(csv_path: str):
    """Load WASB ball detections from CSV."""
    rows = list(csv.DictReader(open(csv_path)))
    data = []
    for r in rows:
        if r['visible'] == '1' and r['x'] and r['y']:
            data.append((int(r['frame']), float(r['x']), float(r['y'])))
    frames = np.array([d[0] for d in data])
    xs = np.array([d[1] for d in data])
    ys = np.array([d[2] for d in data])
    return frames, xs, ys


def compute_kinematics(frames, xs, ys, smooth_window=7):
    """Compute smoothed speed and vertical velocity."""
    speed = np.zeros(len(frames))
    vy = np.zeros(len(frames))
    for i in range(1, len(frames)):
        df = frames[i] - frames[i - 1]
        if 0 < df <= 3:
            dx = (xs[i] - xs[i - 1]) / df
            dy = (ys[i] - ys[i - 1]) / df
            speed[i] = np.sqrt(dx * dx + dy * dy)
            vy[i] = dy
    speed_smooth = uniform_filter1d(speed, size=smooth_window)
    vy_smooth = uniform_filter1d(vy, size=5)
    return speed_smooth, vy_smooth


def segment_rallies(frames, xs, ys, speed_smooth, vy_smooth,
                    low_speed_thresh=2.0, min_rally_frames=20,
                    gap_thresh=10):
    """Segment continuous ball detections into individual rallies.

    A rally boundary is detected when:
    - Ball speed drops below threshold for several frames (dead ball)
    - Detection gap exceeds gap_thresh frames
    """
    is_active = speed_smooth >= low_speed_thresh

    rallies = []
    in_rally = False
    rally_start = 0

    for i in range(len(frames)):
        if is_active[i] and not in_rally:
            rally_start = i
            in_rally = True
        elif not is_active[i] and in_rally:
            rally_end = i - 1
            if frames[rally_end] - frames[rally_start] >= min_rally_frames:
                rallies.append((rally_start, rally_end))
            in_rally = False
        if in_rally and i > 0 and frames[i] - frames[i - 1] > gap_thresh:
            rally_end = i - 1
            if frames[rally_end] - frames[rally_start] >= min_rally_frames:
                rallies.append((rally_start, rally_end))
            rally_start = i

    if in_rally:
        rally_end = len(frames) - 1
        if frames[rally_end] - frames[rally_start] >= min_rally_frames:
            rallies.append((rally_start, rally_end))

    return rallies


def find_ground_contact_frames(vy_smooth, ys, near_y_thresh=350, far_y_thresh=220):
    """Find bounce frames where ball contacts the ground.

    Near-side bounce: vy positive→negative at high Y (ball descending then ascending)
    Far-side bounce: vy negative→positive at low Y
    """
    bounces = []
    for i in range(2, len(vy_smooth) - 2):
        if vy_smooth[i - 1] > 0.3 and vy_smooth[i + 1] < -0.3 and ys[i] > near_y_thresh:
            bounces.append(i)
        elif vy_smooth[i - 1] < -0.3 and vy_smooth[i + 1] > 0.3 and ys[i] < far_y_thresh:
            bounces.append(i)
    return sorted(set(bounces))


def trim_rally_to_ground(rally_start, rally_end, vy_smooth, ys,
                         near_y_thresh=350, far_y_thresh=220):
    """Trim rally so first and last frames are at ground contact (bounce)."""
    seg_vy = vy_smooth[rally_start:rally_end + 1]
    seg_y = ys[rally_start:rally_end + 1]

    bounces = find_ground_contact_frames(seg_vy, seg_y, near_y_thresh, far_y_thresh)

    if len(bounces) < 2:
        return rally_start, rally_end, False

    first_bounce = bounces[0] + rally_start
    last_bounce = bounces[-1] + rally_start

    if last_bounce - first_bounce < 15:
        return rally_start, rally_end, False

    return first_bounce, last_bounce, True


def load_calibrated_camera(cam_path: str):
    """Load calibrated camera parameters."""
    cam = json.load(open(cam_path))
    intrinsics = np.array(cam["intrinsics"], dtype=np.float32)
    extrinsic = np.array(cam["extrinsic"], dtype=np.float32)
    return intrinsics, extrinsic, cam


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Load detections ---
    print(f"[1/4] Loading WASB detections from {args.wasb_csv} ...")
    frames, xs, ys = load_wasb_detections(args.wasb_csv)
    print(f"    {len(frames)} detections, frame range {frames[0]}-{frames[-1]}")

    speed_smooth, vy_smooth = compute_kinematics(frames, xs, ys)

    # --- Step 2: Segment into rallies ---
    print("[2/4] Segmenting into rallies ...")
    raw_rallies = segment_rallies(frames, xs, ys, speed_smooth, vy_smooth,
                                  low_speed_thresh=args.speed_thresh,
                                  min_rally_frames=args.min_rally_frames)
    print(f"    Found {len(raw_rallies)} raw rally segments")

    rallies = []
    for i, (s, e) in enumerate(raw_rallies):
        ts, te, trimmed = trim_rally_to_ground(s, e, vy_smooth, ys)
        rally_frames = frames[ts:te + 1]
        rally_xs = xs[ts:te + 1]
        rally_ys = ys[ts:te + 1]
        status = "trimmed" if trimmed else "untrimmed"
        print(f"    Rally {i}: frames {rally_frames[0]}-{rally_frames[-1]} "
              f"({len(rally_frames)} frames, {status}), "
              f"y=[{rally_ys.min():.0f},{rally_ys.max():.0f}]")
        rallies.append({
            "idx": i,
            "start": ts,
            "end": te,
            "frames": rally_frames,
            "uv": np.column_stack([rally_xs, rally_ys]),
            "trimmed": trimmed,
        })

    # --- Step 3: Camera + LSTM inference ---
    print(f"[3/4] Running LSTM inference ...")
    intrinsics, extrinsic, cam_meta = load_calibrated_camera(args.camera_json)
    convention = cam_meta.get("convention", "opengl")
    if "OpenGL" in convention or "opengl" in convention:
        convention = "opengl"
    else:
        convention = "opencv"
    print(f"    Camera: fx={intrinsics[0]:.0f}, convention={convention}")

    net = WhereIsTheBall(hidden=64).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(state["model_state"])
    net.eval()
    print(f"    Model: epoch {state.get('epoch', '?')}")

    results = []
    for rally in rallies:
        uv = rally["uv"]
        P = pixel_to_plane_points(uv, intrinsics, extrinsic, convention=convention)

        if np.isnan(P).any():
            nan_frac = np.isnan(P).any(axis=1).mean()
            print(f"    Rally {rally['idx']}: {nan_frac:.0%} NaN plane-points — skipping")
            continue

        P_t = torch.from_numpy(P).unsqueeze(0).to(device)
        lengths = torch.tensor([len(P)], dtype=torch.long)

        with torch.no_grad():
            out = net(P_t, lengths=lengths)

        xyz = out["xyz_final"][0].cpu().numpy()
        eps = out["eps"][0].cpu().numpy().squeeze(-1)
        h = out["h_refined"][0].cpu().numpy().squeeze(-1)

        rally["xyz"] = xyz
        rally["eps"] = eps
        rally["height"] = h
        rally["P"] = P
        results.append(rally)

        eot_count = (eps > 0.5).sum()
        print(f"    Rally {rally['idx']}: {len(uv)} frames → "
              f"x=[{xyz[:, 0].min():.1f},{xyz[:, 0].max():.1f}] "
              f"y=[{xyz[:, 1].min():.2f},{xyz[:, 1].max():.2f}] "
              f"z=[{xyz[:, 2].min():.1f},{xyz[:, 2].max():.1f}] "
              f"eot_peaks={eot_count}")

    if not results:
        print("No valid rally segments. Check camera calibration and detection quality.")
        return

    # --- Step 4: Output ---
    print(f"[4/4] Generating outputs ...")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 4a: trajectory plots — side view (Z-Y) and top-down (X-Z) with court/net
    NET_HEIGHT = 1.07
    COURT_HALF = 11.885
    SINGLES_HALF_W = 4.115
    DOUBLES_HALF_W = 5.485

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Rally-Segmented 3D Ball Trajectories", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))

    # --- Side view (Z-Y) ---
    ax = axes[0]
    ax.plot([0, 0], [0, NET_HEIGHT], color="gray", linewidth=2, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(-COURT_HALF, color="gray", linewidth=0.5, alpha=0.3, linestyle="--")
    ax.axvline(COURT_HALF, color="gray", linewidth=0.5, alpha=0.3, linestyle="--")
    for i, res in enumerate(results):
        xyz = res["xyz"]
        c = colors[i % len(colors)]
        label = f"rally {res['idx']} (f{res['frames'][0]}-{res['frames'][-1]}, {len(res['frames'])}f)"
        ax.plot(xyz[:, 2], xyz[:, 1], "-", color=c, label=label, linewidth=1.5, alpha=0.8)
        ax.scatter(xyz[0, 2], xyz[0, 1], color=c, s=30, marker="o", zorder=4)
    ax.set_xlabel("z (m, along court)")
    ax.set_ylabel("y (m, height)")
    ax.set_title("Side view z-y")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-0.5, 4.0)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # --- Top-down (X-Z) ---
    ax = axes[1]
    court_x = [-DOUBLES_HALF_W, DOUBLES_HALF_W, DOUBLES_HALF_W, -DOUBLES_HALF_W, -DOUBLES_HALF_W]
    court_z = [-COURT_HALF, -COURT_HALF, COURT_HALF, COURT_HALF, -COURT_HALF]
    ax.plot(court_x, court_z, color="gray", linewidth=1, alpha=0.5)
    singles_x = [-SINGLES_HALF_W, SINGLES_HALF_W, SINGLES_HALF_W, -SINGLES_HALF_W, -SINGLES_HALF_W]
    ax.plot(singles_x, court_z, color="gray", linewidth=0.5, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    for i, res in enumerate(results):
        xyz = res["xyz"]
        c = colors[i % len(colors)]
        label = f"rally {res['idx']}"
        ax.plot(xyz[:, 0], xyz[:, 2], "-", color=c, label=label, linewidth=1.5, alpha=0.8)
        ax.scatter(xyz[0, 0], xyz[0, 2], color=c, s=30, marker="o", zorder=4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Top-down x-z")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "rally_segmented_3d.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    Saved {plot_path}")

    # 4b: Three.js viewer JSON
    viewer_data = {}
    for i, res in enumerate(results):
        xyz = res["xyz"]
        viewer_data[str(i)] = {
            "pred_refined": xyz.tolist(),
            "pred_unrefined": xyz.tolist(),
            "gt": None,
        }
    viewer_path = os.path.join(out_dir, "rally_segmented_viewer.json")
    with open(viewer_path, "w") as f:
        f.write("var data = ")
        json.dump(viewer_data, f, indent=2)
        f.write(";\n")
    print(f"    Saved {viewer_path}")

    # 4c: Detailed JSON with all metadata
    detail_path = os.path.join(out_dir, "rally_segmented_detail.json")
    export = {
        "wasb_csv": args.wasb_csv,
        "camera": cam_meta,
        "model_ckpt": args.ckpt,
        "model_epoch": state.get("epoch"),
        "num_rallies": len(results),
        "rallies": [],
    }
    for res in results:
        export["rallies"].append({
            "rally_idx": res["idx"],
            "frame_range": [int(res["frames"][0]), int(res["frames"][-1])],
            "num_frames": len(res["frames"]),
            "trimmed_to_ground": res["trimmed"],
            "xyz": res["xyz"].tolist(),
            "eps": res["eps"].tolist(),
            "height": res["height"].tolist(),
            "uv": res["uv"].tolist(),
        })
    with open(detail_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"    Saved {detail_path}")

    total_frames = sum(len(r["frames"]) for r in results)
    print(f"\nDone! {len(results)} rallies, {total_frames} total frames with 3D output.")


def main():
    p = argparse.ArgumentParser(description="Rally-segmented 3D ball trajectory inference")
    p.add_argument("--wasb_csv", required=True, help="Path to WASB ball detections CSV")
    p.add_argument("--camera_json", required=True, help="Path to calibrated camera JSON")
    p.add_argument("--ckpt", default="LSTM_Model/checkpoints_combined/best.pt")
    p.add_argument("--out_dir", default="LSTM_Model/inference_output/segmented")
    p.add_argument("--speed_thresh", type=float, default=2.0,
                   help="Speed threshold for dead-ball detection (px/frame)")
    p.add_argument("--min_rally_frames", type=int, default=20,
                   help="Minimum frames for a valid rally")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
