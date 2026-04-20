#!/usr/bin/env python3
"""Convert the paper's downloaded synthetic tennis trajectories
(paper_data/tennis_synthetic.json — 3D xyz only) into the per-sequence
`.npz` format the paper's LSTM pipeline expects:

    seq_{idx:05d}.npz  with keys
      uv          (L, 2) float32   — projected pixel track
      xyz         (L, 3) float32   — 3D ground truth (y=0 ground, +y up)
      eot         (L,)   uint8     — 1 just before each hit/bounce
      intrinsics  (3,)   float32   — (f, p_x, p_y)
      extrinsic   (4, 4) float32   — E (world→camera, OpenGL convention)

Since the paper only published 3D xyz, we synthesize a broadcast-style camera
and project xyz → uv ourselves. EoT is auto-detected from y-minima (bounces)
and speed discontinuities (mid-air hits).
"""
from __future__ import annotations
import argparse
import ast
import json
import pathlib
import re

import numpy as np


# -------------------- paper JSON parsing --------------------
def parse_paper_js(path: pathlib.Path) -> dict[str, list[list[float]]]:
    """Paper file is a JS-style dict literal: {0: {"gt": [...]}, ...}."""
    txt = path.read_text().strip()
    m = re.search(r'=\s*(.*?)\s*;?\s*$', txt, re.S)
    src = m.group(1) if m else txt
    try:
        d = ast.literal_eval(src)
    except Exception:
        src2 = re.sub(r'(\s|,|{)(\d+)\s*:', r'\1"\2":', src).replace("'", '"')
        d = json.loads(src2)
    return {str(k): np.asarray(v["gt"], dtype=np.float32) for k, v in d.items()}


# -------------------- camera setup --------------------
def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenGL-style world→camera matrix (camera looks down -Z in cam space)."""
    f = (target - eye).astype(np.float64)
    f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, -f], axis=0)             # 3x3 rotation
    t = -R @ eye
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R
    E[:3,  3] = t
    return E


def make_broadcast_camera(width=1280, height=720, fov_y_deg=30.0,
                           eye=(0.0, 8.0, -16.0), look=(0.0, 1.5, 0.0)):
    """High behind-baseline camera, like the paper's TrackNet broadcast view."""
    fy = (height * 0.5) / np.tan(np.radians(fov_y_deg) * 0.5)
    fx = fy
    cx, cy = width * 0.5, height * 0.5
    E = look_at(np.array(eye, dtype=np.float64),
                np.array(look, dtype=np.float64),
                np.array([0.0, 1.0, 0.0]))
    return {
        "width": int(width), "height": int(height),
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "extrinsic": E,                                 # 4x4 world→camera
        "intrinsics": np.array([fx, cx, cy], dtype=np.float32),  # paper's (f, p_x, p_y)
    }


# -------------------- projection --------------------
def project(xyz: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points → Nx2 pixel coords. Returns (uv, z_cam).
    Camera looks down -Z, so visible points have z_cam < 0."""
    E = cam["extrinsic"]
    homog = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    p_cam = homog @ E.T
    z = p_cam[:, 2]
    u = cam["cx"] + cam["fx"] * (p_cam[:, 0] / -z)
    v = cam["cy"] - cam["fy"] * (p_cam[:, 1] / -z)
    return np.stack([u, v], axis=1).astype(np.float32), z


# -------------------- EoT detection --------------------
def detect_eot(xyz: np.ndarray, dt: float = 0.02,
               bounce_y_thresh: float = 0.4,
               speed_jump_sigma: float = 4.0) -> np.ndarray:
    """Mark frame BEFORE each bounce (y-minimum near ground) and each mid-air
    velocity discontinuity (player hit). Output is 1-indexed-aligned: eot[i]=1
    means frame i is the last frame of a trajectory; the next frame begins a new arc."""
    n = len(xyz)
    eot = np.zeros(n, dtype=np.uint8)
    if n < 4:
        return eot

    # Velocity (forward differences, cell N-1)
    v = np.diff(xyz, axis=0) / dt                       # (n-1, 3)

    # --- bounces: vy crosses from negative to positive while ball is low ---
    for i in range(1, len(v)):
        if v[i-1, 1] < 0 and v[i, 1] > 0 and xyz[i, 1] < bounce_y_thresh:
            eot[max(0, i - 1)] = 1

    # --- mid-air hits: large jump in velocity vector (Δv is an outlier) ---
    dv = np.linalg.norm(np.diff(v, axis=0), axis=1)     # (n-2,)
    if len(dv) > 5:
        med = float(np.median(dv))
        mad = float(np.median(np.abs(dv - med)) + 1e-6)
        thresh = med + speed_jump_sigma * 1.4826 * mad
        for i, d in enumerate(dv):
            if d > thresh:
                eot[max(0, i)] = 1
    return eot


# -------------------- main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paper_json", required=True, type=pathlib.Path)
    p.add_argument("--out_dir",    required=True, type=pathlib.Path)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov_y_deg", type=float, default=30.0)
    p.add_argument("--cam_eye", type=float, nargs=3, default=(0.0, 8.0, -16.0))
    p.add_argument("--cam_look", type=float, nargs=3, default=(0.0, 1.5, 0.0))
    p.add_argument("--dt", type=float, default=0.02,
                   help="frame timestep (s). Paper uses 50fps → 0.02.")
    p.add_argument("--shift_to_positive_z", action="store_true",
                   help="Translate all rallies uniformly so min(z) >= z_margin. "
                        "Required for the paper's p_v.y > 0 convention to hold "
                        "when the original data spans z<0 (camera-side balls).")
    p.add_argument("--z_margin", type=float, default=1.0,
                   help="Smallest z value after shift (meters above the vertical plane).")
    args = p.parse_args()

    rallies = parse_paper_js(args.paper_json)

    # --- optional: shift all rallies so they sit entirely on +z side ---
    z_shift = 0.0
    if args.shift_to_positive_z:
        global_min_z = min(float(xyz[:, 2].min()) for xyz in rallies.values())
        z_shift = -global_min_z + args.z_margin
        print(f"[shift] global min z = {global_min_z:.3f}  →  shifting all rallies by Δz = {z_shift:+.3f}")
        for k in rallies:
            rallies[k] = rallies[k].copy()
            rallies[k][:, 2] += z_shift

    cam = make_broadcast_camera(args.width, args.height, args.fov_y_deg,
                                 tuple(args.cam_eye), tuple(args.cam_look))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    n_eot_total = 0
    for idx, (k, xyz) in enumerate(sorted(rallies.items(), key=lambda kv: int(kv[0]))):
        uv, z_cam = project(xyz, cam)
        # Paper precondition: visible & p_v.y > 0. Check visibility (z_cam<0)
        # and that the ground-plane intersection exists for the lift step.
        if (z_cam >= 0).any():
            skipped += 1
            continue
        eot = detect_eot(xyz, dt=args.dt)
        n_eot_total += int(eot.sum())
        out_path = args.out_dir / f"seq_{idx:05d}.npz"
        np.savez(out_path,
                 uv=uv,
                 xyz=xyz.astype(np.float32),
                 eot=eot,
                 intrinsics=cam["intrinsics"],
                 extrinsic=cam["extrinsic"].astype(np.float32))
        written += 1

    # Also write a single camera.json next to the npz files so consumers can
    # see the camera config without unpickling.
    (args.out_dir / "camera.json").write_text(json.dumps({
        "width": cam["width"], "height": cam["height"],
        "fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"],
        "extrinsic": cam["extrinsic"].tolist(),
        "fov_y_deg": args.fov_y_deg,
        "eye": list(args.cam_eye), "look": list(args.cam_look),
        "convention": "OpenGL view (cam looks down -Z, +Y up)",
        "z_shift_applied": z_shift,
        "z_margin": args.z_margin,
    }, indent=2))

    print(f"wrote {written} sequences  ({skipped} skipped: behind-camera frames)")
    print(f"total EoT flags: {n_eot_total}  → mean {n_eot_total / max(written,1):.2f} per rally")
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
