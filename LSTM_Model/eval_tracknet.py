#!/usr/bin/env python3
"""Evaluate LSTM model on real TrackNet tennis dataset (paper Stage 2).

For each game:
1. Calibrate camera using TennisCourtDetector on first frame
2. For each clip (rally): extract 2D ball positions from Label.csv
3. Compute plane-points parameterization
4. Run LSTM inference
5. Compute reprojection error and landing position error at bounce frames

Usage:
    python eval_tracknet.py --ckpt checkpoints_trimmed/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _parent)
sys.path.insert(0, os.path.join(_parent, "TennisCourtDetector"))

from LSTM_Model.data.parameterization import pixel_to_plane_points
from LSTM_Model.pipeline import WhereIsTheBall


# ---- Camera calibration via TennisCourtDetector ----

COURT_3D_POINTS = np.array([
    [-5.485, 0, 11.885],   # 0: baseline_top left
    [5.485, 0, 11.885],    # 1: baseline_top right
    [-5.485, 0, -11.885],  # 2: baseline_bottom left
    [5.485, 0, -11.885],   # 3: baseline_bottom right
    [-4.115, 0, 11.885],   # 4: left_inner top
    [-4.115, 0, -11.885],  # 5: left_inner bottom
    [4.115, 0, 11.885],    # 6: right_inner top
    [4.115, 0, -11.885],   # 7: right_inner bottom
    [-4.115, 0, 6.401],    # 8: top_inner left
    [4.115, 0, 6.401],     # 9: top_inner right
    [-4.115, 0, -6.401],   # 10: bottom_inner left
    [4.115, 0, -6.401],    # 11: bottom_inner right
    [0, 0, 6.401],         # 12: middle_line top
    [0, 0, -6.401],        # 13: middle_line bottom
], dtype=np.float64)


def calibrate_camera_from_image(image_path: str, model_path: str, device: str = "cuda",
                                use_homography: bool = True):
    """Detect court keypoints, optionally refine via homography, and solve PnP."""
    from tracknet import BallTrackerNet
    from postprocess import postprocess, refine_kps

    model = BallTrackerNet(out_channels=15)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.eval()

    image = cv2.imread(image_path)
    if image is None:
        return None
    h, w = image.shape[:2]
    img = cv2.resize(image, (640, 360))
    inp = (img.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0)

    out = model(inp.float().to(dev))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()

    # postprocess(scale=2) maps from 320x180 heatmap to 1280x720.
    # For images at other resolutions, adjust from 1280x720 to actual size.
    post_scale_x = w / 1280.0
    post_scale_y = h / 720.0

    all_kps_14 = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if x_pred is not None and y_pred is not None:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            all_kps_14.append((x_pred * post_scale_x, y_pred * post_scale_y))
        else:
            all_kps_14.append((None, None))

    detected_count = sum(1 for p in all_kps_14 if p[0] is not None)
    if detected_count < 4:
        return None

    if use_homography and detected_count >= 4:
        from homography import get_trans_matrix
        from court_reference import CourtReference
        matrix = get_trans_matrix(all_kps_14)
        if matrix is not None:
            court_ref = CourtReference()
            refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))
            corrected = cv2.perspectiveTransform(refer_kps, matrix)
            points_2d = []
            points_3d = []
            for i in range(14):
                pt = corrected[i].flatten()
                if 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                    points_2d.append([float(pt[0]), float(pt[1])])
                    points_3d.append(COURT_3D_POINTS[i])
        else:
            points_2d = []
            points_3d = []
            for i, (px, py) in enumerate(all_kps_14):
                if px is not None:
                    points_2d.append([px, py])
                    points_3d.append(COURT_3D_POINTS[i])
    else:
        points_2d = []
        points_3d = []
        for i, (px, py) in enumerate(all_kps_14):
            if px is not None:
                points_2d.append([px, py])
                points_3d.append(COURT_3D_POINTS[i])

    if len(points_2d) < 6:
        return None

    points_2d = np.array(points_2d, dtype=np.float64)
    points_3d = np.array(points_3d, dtype=np.float64)

    best_err = float("inf")
    best_result = None
    best_rvec, best_tvec, best_K = None, None, None
    for focal in range(800, 3500, 50):
        K = np.array([[focal, 0, w / 2.0],
                       [0, focal, h / 2.0],
                       [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K, None,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
        err = np.mean(np.sqrt(((proj.squeeze() - points_2d) ** 2).sum(axis=1)))
        if err < best_err:
            best_err = err
            best_rvec, best_tvec, best_K = rvec, tvec, K
            R, _ = cv2.Rodrigues(rvec)
            flip = np.diag([1, -1, -1]).astype(np.float64)
            R_gl = flip @ R
            t_gl = flip @ tvec.flatten()
            E = np.eye(4, dtype=np.float64)
            E[:3, :3] = R_gl
            E[:3, 3] = t_gl
            E[:3, 1] *= -1  # fix Y-reflection ambiguity: PnP places camera below ground
            cam_pos = -E[:3, :3].T @ E[:3, 3]
            best_result = {
                "fx": float(focal), "fy": float(focal),
                "cx": w / 2.0, "cy": h / 2.0,
                "intrinsics": np.array([focal, w / 2.0, h / 2.0], dtype=np.float32),
                "extrinsic": E.astype(np.float32),
                "reprojection_error": best_err,
                "camera_position": cam_pos.tolist(),
                "num_keypoints": len(points_2d),
                "width": w, "height": h,
                "_raw_kps": all_kps_14,
                "_rvec": best_rvec, "_tvec": best_tvec, "_K": best_K,
            }

    return best_result


COURT_LINES = [
    (0, 1), (2, 3), (0, 2), (1, 3),
    (4, 5), (6, 7),
    (4, 6), (8, 9), (10, 11), (5, 7),
    (12, 13),
]


def save_calibration_vis(image_path: str, cam: dict, out_path: str):
    """Draw PnP-reprojected court model and raw detections on the image."""
    image = cv2.imread(image_path)
    if image is None:
        return
    vis = image.copy()
    h, w = image.shape[:2]
    raw_kps = cam["_raw_kps"]
    rvec, tvec, K = cam["_rvec"], cam["_tvec"], cam["_K"]

    proj_all, _ = cv2.projectPoints(COURT_3D_POINTS, rvec, tvec, K, None)
    proj_pts = proj_all.squeeze()

    for i, j in COURT_LINES:
        pt1, pt2 = proj_pts[i], proj_pts[j]
        cv2.line(vis, (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
    for i, pt in enumerate(proj_pts):
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

    detected = 0
    for i, (px, py) in enumerate(raw_kps):
        if px is not None:
            detected += 1
            cv2.circle(vis, (int(px), int(py)), 4, (255, 0, 0), -1)
            cv2.putText(vis, str(i), (int(px) + 6, int(py) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    title = (f"Blue: raw detected ({detected}/14), Green: PnP reprojected "
             f"(fx={cam['fx']:.0f}, err={cam['reprojection_error']:.2f}px)")
    cv2.putText(vis, title, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(out_path, vis)


def load_clip_labels(clip_dir: str):
    """Load ball positions and status from Label.csv."""
    label_path = os.path.join(clip_dir, "Label.csv")
    if not os.path.exists(label_path):
        return None

    frames = []
    with open(label_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vis = int(row["visibility"])
            x = float(row["x-coordinate"]) if row["x-coordinate"] else 0
            y = float(row["y-coordinate"]) if row["y-coordinate"] else 0
            status = row.get("status", "0").strip()
            status = int(status) if status else 0
            frames.append({
                "filename": row["file name"],
                "visible": vis > 0,
                "x": x, "y": y,
                "status": status,  # 0=flying, 1=bounce, 2=hit
            })
    return frames


def project_3d_to_2d(xyz, intrinsics, extrinsic):
    """Project 3D points to 2D pixels (OpenGL convention)."""
    fx, cx, cy = intrinsics
    E = extrinsic.astype(np.float64)
    ones = np.ones((len(xyz), 1))
    homog = np.concatenate([xyz, ones], axis=1)
    p_cam = (E @ homog.T).T
    z_cam = p_cam[:, 2]
    u = cx + fx * (p_cam[:, 0] / (-z_cam))
    v = cy - fx * (p_cam[:, 1] / (-z_cam))
    return np.stack([u, v], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints_trimmed/best.pt")
    p.add_argument("--tracknet_dir", default="../TrackNet/datasets/trackNet/Dataset")
    p.add_argument("--court_model", default="../TennisCourtDetector/model_best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="inference_output/tracknet_eval")
    p.add_argument("--games", type=str, default=None,
                   help="Comma-separated game numbers to evaluate (e.g., '1,2,3'). Default: all.")
    p.add_argument("--vis", action="store_true",
                   help="Save calibration visualization PNGs")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load LSTM model
    print("[1/3] Loading LSTM model ...")
    net = WhereIsTheBall(hidden=64).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(state["model_state"])
    net.eval()
    print(f"    Loaded epoch {state.get('epoch', '?')}")

    # Find all games
    games = sorted([d for d in os.listdir(args.tracknet_dir)
                    if d.startswith("game") and os.path.isdir(os.path.join(args.tracknet_dir, d))])
    if args.games:
        selected = set(f"game{g}" for g in args.games.split(","))
        games = [g for g in games if g in selected]
    print(f"    Games to evaluate: {games}")

    # Results accumulators
    all_reproj_errors = []
    all_landing_errors = []
    clip_results = []
    skipped_clips = 0
    total_clips = 0

    print("\n[2/3] Processing games ...")
    for game in games:
        game_dir = os.path.join(args.tracknet_dir, game)
        clips = sorted([d for d in os.listdir(game_dir)
                        if d.startswith("Clip") and os.path.isdir(os.path.join(game_dir, d))])

        # Calibrate camera using first frame of first clip
        first_clip = os.path.join(game_dir, clips[0])
        first_frame = os.path.join(first_clip, "0000.jpg")
        if not os.path.exists(first_frame):
            frames_in_clip = sorted([f for f in os.listdir(first_clip) if f.endswith(".jpg")])
            if frames_in_clip:
                first_frame = os.path.join(first_clip, frames_in_clip[0])
            else:
                print(f"  {game}: no frames found, skipping")
                continue

        print(f"\n  {game}: calibrating camera from {os.path.basename(first_frame)} ...")
        cam = calibrate_camera_from_image(first_frame, args.court_model, args.device)
        if cam is None:
            print(f"  {game}: camera calibration failed, skipping")
            skipped_clips += len(clips)
            continue
        print(f"    Camera: fx={cam['fx']:.0f}, reproj={cam['reprojection_error']:.2f}px, "
              f"keypoints={cam['num_keypoints']}")

        if args.vis:
            vis_dir = os.path.join(args.out_dir, "calibration_vis")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"{game}_calibration.png")
            save_calibration_vis(first_frame, cam, vis_path)
            print(f"    Saved visualization -> {vis_path}")

        intrinsics = cam["intrinsics"]
        extrinsic = cam["extrinsic"]

        for clip_name in clips:
            total_clips += 1
            clip_dir = os.path.join(game_dir, clip_name)
            labels = load_clip_labels(clip_dir)
            if labels is None:
                skipped_clips += 1
                continue

            # Extract visible ball positions
            uv = []
            statuses = []
            for lbl in labels:
                if lbl["visible"] and lbl["x"] > 0 and lbl["y"] > 0:
                    uv.append([lbl["x"], lbl["y"]])
                    statuses.append(lbl["status"])
                else:
                    uv.append([np.nan, np.nan])
                    statuses.append(0)

            uv = np.array(uv, dtype=np.float32)
            statuses = np.array(statuses)

            # Find valid (non-nan) frames
            valid_mask = ~np.isnan(uv[:, 0])
            valid_count = valid_mask.sum()
            if valid_count < 10:
                skipped_clips += 1
                continue

            # Use only valid frames for inference
            valid_idx = np.where(valid_mask)[0]
            uv_valid = uv[valid_idx]

            # Compute plane-points
            P = pixel_to_plane_points(uv_valid, intrinsics, extrinsic, convention="opengl")
            if np.isnan(P).any():
                nan_frac = np.isnan(P).any(axis=1).mean()
                if nan_frac > 0.5:
                    skipped_clips += 1
                    continue
                good = ~np.isnan(P).any(axis=1)
                P = P[good]
                uv_valid = uv_valid[good]
                valid_idx = valid_idx[good]
                statuses_valid = statuses[valid_idx]
            else:
                statuses_valid = statuses[valid_idx]

            if len(P) < 10:
                skipped_clips += 1
                continue

            # LSTM inference
            P_t = torch.from_numpy(P).unsqueeze(0).to(device)
            lengths = torch.tensor([len(P)], dtype=torch.long)
            with torch.no_grad():
                out = net(P_t, lengths=lengths)
            xyz = out["xyz_final"][0].cpu().numpy()

            # Reprojection error
            uv_proj = project_3d_to_2d(xyz, intrinsics, extrinsic)
            reproj_err = np.sqrt(((uv_proj - uv_valid) ** 2).sum(axis=1))
            all_reproj_errors.extend(reproj_err.tolist())

            # Landing error at bounce frames (status=1)
            bounce_mask = statuses_valid == 1
            if bounce_mask.any():
                bounce_xyz = xyz[bounce_mask]
                landing_err = np.abs(bounce_xyz[:, 1])  # distance from ground (y=0)
                all_landing_errors.extend(landing_err.tolist())

            clip_results.append({
                "game": game,
                "clip": clip_name,
                "num_frames": int(valid_count),
                "reproj_mean": float(reproj_err.mean()),
                "reproj_median": float(np.median(reproj_err)),
                "num_bounces": int(bounce_mask.sum()),
                "h_max": float(xyz[:, 1].max()),
                "h_min": float(xyz[:, 1].min()),
            })

            print(f"    {clip_name}: {valid_count} frames, "
                  f"reproj={reproj_err.mean():.1f}px (med={np.median(reproj_err):.1f}), "
                  f"bounces={bounce_mask.sum()}, h=[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}]m")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"[3/3] EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total clips: {total_clips}, evaluated: {len(clip_results)}, skipped: {skipped_clips}")

    if all_reproj_errors:
        err = np.array(all_reproj_errors)
        print(f"\nReprojection error ({len(err)} frames):")
        print(f"  Mean:   {err.mean():.2f} px")
        print(f"  Median: {np.median(err):.2f} px")
        print(f"  Std:    {err.std():.2f} px")
        print(f"  <5px:   {100*(err<5).mean():.1f}%")
        print(f"  <10px:  {100*(err<10).mean():.1f}%")
        print(f"  <20px:  {100*(err<20).mean():.1f}%")

    if all_landing_errors:
        le = np.array(all_landing_errors)
        print(f"\nLanding height error at bounce frames ({len(le)} bounces):")
        print(f"  Mean:   {le.mean():.3f} m")
        print(f"  Median: {np.median(le):.3f} m")
        print(f"  Std:    {le.std():.3f} m")
        print(f"  Paper reference: ~0.63 m (Table 4)")

    # Save detailed results
    results = {
        "model_epoch": state.get("epoch"),
        "total_clips": total_clips,
        "evaluated_clips": len(clip_results),
        "skipped_clips": skipped_clips,
        "reproj_mean_px": float(np.mean(all_reproj_errors)) if all_reproj_errors else None,
        "reproj_median_px": float(np.median(all_reproj_errors)) if all_reproj_errors else None,
        "landing_mean_m": float(np.mean(all_landing_errors)) if all_landing_errors else None,
        "landing_median_m": float(np.median(all_landing_errors)) if all_landing_errors else None,
        "clips": clip_results,
    }
    out_path = os.path.join(args.out_dir, "tracknet_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
