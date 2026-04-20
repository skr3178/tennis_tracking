#!/usr/bin/env python3
"""Convert OUR Unity-generated dataset (frames.csv + camera.json per episode)
into the same per-sequence .npz format paper_to_npz.py emits.

Trims each episode so the first and last frames are at ground contact (y ≈ 0),
matching the paper's assumption that h_f0 = 0 and h_bN = 0.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import numpy as np


def find_ground_contacts(y: np.ndarray, vy: np.ndarray,
                          thresh: float = 0.05) -> np.ndarray:
    """Return frame indices where ball is at ground contact.

    Requires y < thresh AND ball moving downward or near-stationary (vy <= 1.0),
    to avoid catching mid-bounce frames with upward velocity.
    """
    near_ground = y < thresh
    not_rising = vy <= 1.0
    return np.where(near_ground & not_rising)[0]


def convert(ep_dir: pathlib.Path, ground_thresh: float = 0.05,
            min_len: int = 20) -> dict | None:
    f = np.genfromtxt(ep_dir / "frames.csv", delimiter=",", names=True)
    cam = json.loads((ep_dir / "camera.json").read_text())

    xyz = np.stack([f["x"], f["y"], f["z"]], axis=1).astype(np.float32)
    uv  = np.stack([f["u"], f["v"]], axis=1).astype(np.float32)
    eot = f["eot"].astype(np.uint8)
    vy = f["vy"].astype(np.float32) if "vy" in f.dtype.names else np.zeros(len(xyz))

    gc = find_ground_contacts(xyz[:, 1], vy, thresh=ground_thresh)
    if len(gc) < 2:
        return None

    start, end = gc[0], gc[-1]
    if end - start + 1 < min_len:
        return None

    xyz = xyz[start:end + 1]
    uv = uv[start:end + 1]
    eot = eot[start:end + 1]

    intr = np.array([cam["fx"], cam["cx"], cam["cy"]], dtype=np.float32)
    E = np.asarray(cam["worldToCamera"], dtype=np.float32)
    return {"uv": uv, "xyz": xyz, "eot": eot, "intrinsics": intr, "extrinsic": E}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True, type=pathlib.Path)
    p.add_argument("--out_dir",   required=True, type=pathlib.Path)
    p.add_argument("--ground_thresh", type=float, default=0.05,
                   help="y threshold for ground contact (m)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eps = sorted(d for d in args.split_dir.iterdir()
                 if d.is_dir() and d.name.startswith("ep_"))
    written = 0
    skipped = 0
    lengths = []
    for i, ep in enumerate(eps):
        d = convert(ep, ground_thresh=args.ground_thresh)
        if d is None:
            skipped += 1
            continue
        np.savez(args.out_dir / f"seq_{written:05d}.npz", **d)
        lengths.append(len(d["xyz"]))
        written += 1

    print(f"wrote {written} sequences, skipped {skipped} -> {args.out_dir}")
    if lengths:
        print(f"  length: min={min(lengths)} max={max(lengths)} "
              f"mean={np.mean(lengths):.0f} median={np.median(lengths):.0f}")
        print(f"  first_y range: [{min(np.load(args.out_dir/f'seq_{i:05d}.npz')['xyz'][0,1] for i in range(written)):.4f}, "
              f"{max(np.load(args.out_dir/f'seq_{i:05d}.npz')['xyz'][0,1] for i in range(written)):.4f}]")


if __name__ == "__main__":
    main()
