#!/usr/bin/env python3
"""Standalone PnP Y-ambiguity diagnostic.

Reads the first frame of each TrackNet game, detects court keypoints via
TennisCourtDetector, and solves PnP in three different ways:

  (A) SOLVEPNP_ITERATIVE with focal sweep (what eval_tracknet.py does today)
  (B) SOLVEPNP_IPPE (returns BOTH coplanar solutions, pick each in turn)

For each solution it reports, in the OpenGL/Unity frame used downstream:
    - camera position (cam_pos = -R_gl^T @ t_gl)
    - reprojection error on the court points
    - Y sign of cam_pos

This answers: does ITERATIVE pick cam_pos_y < 0 consistently, and does
IPPE expose a valid cam_pos_y > 0 solution with similar reprojection error?

Does NOT modify any other code. Purely diagnostic.

Usage:
    python LSTM_Model/diagnose_pnp_ambiguity.py \
        --dataset-root TrackNet/datasets/trackNet/Dataset \
        --court-model TennisCourtDetector/model_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


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

GL_FLIP = np.diag([1, -1, -1]).astype(np.float64)


def detect_court_kps(image: np.ndarray, court_model_path: str, device: str):
    """Run TennisCourtDetector on a BGR image, return list of 14 (x,y) or (None,None)."""
    # TennisCourtDetector lives next to LSTM_Model in the repo root.
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    tcd = repo_root / "TennisCourtDetector"
    if str(tcd) not in sys.path:
        sys.path.insert(0, str(tcd))
    from tracknet import BallTrackerNet
    from postprocess import postprocess, refine_kps
    from homography import get_trans_matrix
    from court_reference import CourtReference

    model = BallTrackerNet(out_channels=15)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.load_state_dict(torch.load(court_model_path, map_location=dev))
    model.eval()

    h, w = image.shape[:2]
    img = cv2.resize(image, (640, 360))
    inp = (img.astype(np.float32) / 255.0)
    inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0)
    out = model(inp.float().to(dev))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()

    post_scale_x = w / 1280.0
    post_scale_y = h / 720.0

    all_kps_14 = []
    for k in range(14):
        heatmap = (pred[k] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if x_pred is not None and y_pred is not None:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            all_kps_14.append((x_pred * post_scale_x, y_pred * post_scale_y))
        else:
            all_kps_14.append((None, None))

    # Homography refinement (mirrors eval_tracknet.py behavior).
    detected = sum(1 for p in all_kps_14 if p[0] is not None)
    if detected >= 4:
        matrix = get_trans_matrix(all_kps_14)
        if matrix is not None:
            court_ref = CourtReference()
            refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))
            corrected = cv2.perspectiveTransform(refer_kps, matrix)
            refined = []
            for i in range(14):
                pt = corrected[i].flatten()
                if 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                    refined.append((float(pt[0]), float(pt[1])))
                else:
                    refined.append((None, None))
            return refined
    return all_kps_14


def _to_gl(rvec, tvec):
    """Convert OpenCV (rvec, tvec) to OpenGL/Unity world->camera (R_gl, t_gl, cam_pos)."""
    R, _ = cv2.Rodrigues(rvec)
    R_gl = GL_FLIP @ R
    t_gl = GL_FLIP @ tvec.flatten()
    cam_pos = -R_gl.T @ t_gl
    return R_gl, t_gl, cam_pos


def _reproj_err(points_3d, points_2d, rvec, tvec, K):
    proj, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
    return float(np.mean(np.sqrt(((proj.squeeze() - points_2d) ** 2).sum(axis=1))))


def solve_iterative_sweep(points_3d, points_2d, w, h):
    """Exactly mirrors eval_tracknet.py: focal sweep with SOLVEPNP_ITERATIVE."""
    best = None
    for focal in range(800, 3500, 50):
        K = np.array([[focal, 0, w / 2.0],
                      [0, focal, h / 2.0],
                      [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        err = _reproj_err(points_3d, points_2d, rvec, tvec, K)
        if best is None or err < best["err"]:
            R_gl, t_gl, cam_pos = _to_gl(rvec, tvec)
            best = dict(err=err, focal=float(focal), cam_pos=cam_pos,
                        R_gl=R_gl, t_gl=t_gl)
    return best


def solve_ippe_sweep(points_3d, points_2d, w, h):
    """SOLVEPNP_IPPE returns both coplanar solutions. Sweep focal, keep best of each sign."""
    # solvePnPGeneric returns all candidate solutions for a given method.
    best_posY = None
    best_negY = None
    for focal in range(800, 3500, 50):
        K = np.array([[focal, 0, w / 2.0],
                      [0, focal, h / 2.0],
                      [0, 0, 1]], dtype=np.float64)
        try:
            ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                points_3d, points_2d, K, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            continue
        if not ok:
            continue
        for rvec, tvec in zip(rvecs, tvecs):
            err = _reproj_err(points_3d, points_2d, rvec, tvec, K)
            R_gl, t_gl, cam_pos = _to_gl(rvec, tvec)
            rec = dict(err=err, focal=float(focal), cam_pos=cam_pos,
                       R_gl=R_gl, t_gl=t_gl)
            if cam_pos[1] >= 0:
                if best_posY is None or err < best_posY["err"]:
                    best_posY = rec
            else:
                if best_negY is None or err < best_negY["err"]:
                    best_negY = rec
    return best_posY, best_negY


def _fmt_cam(cam):
    return f"({cam[0]:+.2f}, {cam[1]:+.2f}, {cam[2]:+.2f})"


def diagnose_one(image_path: str, court_model_path: str, device: str, label: str):
    image = cv2.imread(image_path)
    if image is None:
        return {"label": label, "error": f"cannot read {image_path}"}
    h, w = image.shape[:2]

    kps = detect_court_kps(image, court_model_path, device)
    points_2d, points_3d = [], []
    for i, (px, py) in enumerate(kps):
        if px is not None:
            points_2d.append([px, py])
            points_3d.append(COURT_3D_POINTS[i])
    if len(points_2d) < 6:
        return {"label": label, "error": f"only {len(points_2d)} court points detected"}

    points_2d = np.array(points_2d, dtype=np.float64)
    points_3d = np.array(points_3d, dtype=np.float64)

    iterative = solve_iterative_sweep(points_3d, points_2d, w, h)
    ippe_pos, ippe_neg = solve_ippe_sweep(points_3d, points_2d, w, h)

    return {
        "label": label,
        "image": image_path,
        "resolution": [w, h],
        "n_points": int(len(points_2d)),
        "iterative": iterative,
        "ippe_posY": ippe_pos,
        "ippe_negY": ippe_neg,
    }


def print_report(results):
    print("\n" + "=" * 100)
    print(f"{'label':<14}{'iter cam_pos (GL)':<34}{'iter err':>10}"
          f"{'  ':<4}{'IPPE +Y cam':<28}{'+Y err':>8}"
          f"{'  ':<4}{'IPPE -Y cam':<28}{'-Y err':>8}")
    print("=" * 100)
    neg_count = 0
    pos_count = 0
    iter_matches_pos = 0
    iter_matches_neg = 0
    for r in results:
        if "error" in r:
            print(f"{r['label']:<14}  ERROR: {r['error']}")
            continue
        it = r["iterative"]
        p = r["ippe_posY"]
        n = r["ippe_negY"]
        it_sign = "+" if it["cam_pos"][1] >= 0 else "-"
        if it_sign == "+":
            pos_count += 1
        else:
            neg_count += 1
        p_str = _fmt_cam(p["cam_pos"]) if p else "(none)"
        n_str = _fmt_cam(n["cam_pos"]) if n else "(none)"
        p_err = f"{p['err']:.2f}" if p else "-"
        n_err = f"{n['err']:.2f}" if n else "-"
        # Did iterative land on the +Y or -Y basin?
        if p and np.allclose(it["cam_pos"], p["cam_pos"], atol=0.5):
            iter_matches_pos += 1
        if n and np.allclose(it["cam_pos"], n["cam_pos"], atol=0.5):
            iter_matches_neg += 1
        print(f"{r['label']:<14}{_fmt_cam(it['cam_pos']) + ' ' + it_sign:<34}"
              f"{it['err']:>10.2f}"
              f"    {p_str:<28}{p_err:>8}"
              f"    {n_str:<28}{n_err:>8}")

    print("=" * 100)
    total = pos_count + neg_count
    if total == 0:
        print("No successful solves.")
        return
    print(f"\nSummary over {total} frame(s):")
    print(f"  ITERATIVE picked cam_pos_y > 0 : {pos_count}")
    print(f"  ITERATIVE picked cam_pos_y < 0 : {neg_count}")
    print(f"  ITERATIVE landed in IPPE +Y basin : {iter_matches_pos}")
    print(f"  ITERATIVE landed in IPPE -Y basin : {iter_matches_neg}")
    print()
    print("Interpretation:")
    if neg_count == total and neg_count > 0:
        print("  * ITERATIVE is *consistently* picking the below-ground (cam_pos_y < 0) solution.")
        print("  * If IPPE +Y column shows comparable reprojection error (within ~1-2px of -Y),")
        print("    the Y-reflection ambiguity hypothesis holds and switching to IPPE + picking")
        print("    cam_pos_y>0 is the correct, principled fix.")
    elif pos_count == total:
        print("  * ITERATIVE is already above ground. The 48px downstream reprojection error")
        print("    is NOT caused by the Y-mirror ambiguity. Look elsewhere (intrinsics guess,")
        print("    parameterization P convention, or trained-model coordinate frame).")
    else:
        print("  * ITERATIVE is inconsistent across frames — result depends on initialization.")
        print("    IPPE + sign-select would also stabilize this.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="TrackNet/datasets/trackNet/Dataset",
                    help="Root with game1/Clip1/0001.jpg, etc.")
    ap.add_argument("--court-model", default="TennisCourtDetector/model_best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", nargs="*", default=None,
                    help="Explicit list of game names (default: game1..game10).")
    ap.add_argument("--clip", default="Clip1",
                    help="Clip subfolder to use for the first frame.")
    ap.add_argument("--json-out", default=None,
                    help="Optional path to dump full numeric results as JSON.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)
    court_model = Path(args.court_model).resolve()
    if not court_model.exists():
        print(f"ERROR: court model not found: {court_model}", file=sys.stderr)
        sys.exit(1)

    if args.games is None:
        games = [f"game{i}" for i in range(1, 11)]
    else:
        games = args.games

    results = []
    for g in games:
        frame = dataset_root / g / args.clip / "0001.jpg"
        if not frame.exists():
            # try any clip
            alt = next((dataset_root / g).glob("Clip*/0001.jpg"), None)
            if alt is None:
                results.append({"label": g, "error": f"no frame found under {dataset_root / g}"})
                continue
            frame = alt
        print(f"[{g}] diagnosing {frame} ...", flush=True)
        r = diagnose_one(str(frame), str(court_model), args.device, label=g)
        results.append(r)

    print_report(results)

    if args.json_out:
        def _ser(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return str(o)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=_ser)
        print(f"\nWrote full results to {args.json_out}")


if __name__ == "__main__":
    main()
