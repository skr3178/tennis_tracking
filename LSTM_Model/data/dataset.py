"""Variable-length ball-trajectory dataset (paper §C / data spec).

Each sequence file (.npz) has:
    uv          (L, 2) float32   2D ball pixel track
    xyz         (L, 3) float32   GT 3D position (y=0 ground, +y up)
    eot         (L,)   uint8     1 just before each hit, else 0
    intrinsics  (3,)   float32   (f, cx, cy)
    extrinsic   (4, 4) float32   world -> camera

A sibling `camera.json` (optional) provides the ray convention; we default
to OpenGL because rev1 sequences use that convention.

Augmentation: per paper §D recipe, Gaussian noise is added to (u, v) before
parameterization on every training sample.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .parameterization import pixel_to_plane_points


def _load_camera_convention(split_dir: str) -> str:
    cam_path = os.path.join(split_dir, "camera.json")
    if os.path.isfile(cam_path):
        try:
            with open(cam_path) as f:
                conv = json.load(f).get("convention", "")
            if "opencv" in conv.lower():
                return "opencv"
        except Exception:
            pass
    return "opengl"


def _split_indices(n: int, train_frac: float, val_frac: float,
                   test_frac: float, seed: int) -> Dict[str, np.ndarray]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(train_frac * n))
    n_va = int(round(val_frac * n))
    return {
        "train": perm[:n_tr],
        "val":   perm[n_tr:n_tr + n_va],
        "test":  perm[n_tr + n_va:],
    }


class BallTrajectoryDataset(Dataset):
    """Loads all .npz files under `<root>/<split_subdir>/` and returns
    sequences for the requested split. Parameterization is done on the fly
    so that uv-noise augmentation is applied per sample.
    """

    def __init__(self, root: str, split_subdir: str, split: str,
                 train_frac: float = 0.72, val_frac: float = 0.18,
                 test_frac: float = 0.10, seed: int = 0,
                 uv_noise_sigma_px: float = 0.0,
                 clamp_gt_y_nonneg: bool = False):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self.uv_noise_sigma_px = float(uv_noise_sigma_px)
        self.clamp_gt_y_nonneg = bool(clamp_gt_y_nonneg)

        self.split_dir = os.path.join(root, split_subdir)
        self.convention = _load_camera_convention(self.split_dir)

        files = sorted(glob.glob(os.path.join(self.split_dir, "seq_*.npz")))
        if not files:
            raise FileNotFoundError(f"No seq_*.npz under {self.split_dir}")
        idx = _split_indices(len(files), train_frac, val_frac, test_frac, seed)
        self.files: List[str] = [files[i] for i in idx[split]]

        # Cache the raw arrays so __getitem__ is cheap (only re-parameterize).
        self._cache: List[Dict[str, np.ndarray]] = []
        for fn in self.files:
            d = np.load(fn)
            self._cache.append({
                "uv":         d["uv"].astype(np.float32),
                "xyz":        d["xyz"].astype(np.float32),
                "eot":        d["eot"].astype(np.float32),
                "intrinsics": d["intrinsics"].astype(np.float32),
                "extrinsic":  d["extrinsic"].astype(np.float32),
            })

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self._cache[idx]
        uv = d["uv"]
        if self.split == "train" and self.uv_noise_sigma_px > 0:
            uv = uv + np.random.normal(0.0, self.uv_noise_sigma_px,
                                        size=uv.shape).astype(np.float32)
        P = pixel_to_plane_points(uv, d["intrinsics"], d["extrinsic"],
                                   convention=self.convention)
        xyz = d["xyz"]
        if self.clamp_gt_y_nonneg:
            xyz = xyz.copy()
            xyz[:, 1] = np.maximum(xyz[:, 1], 0.0)
        return {
            "P":      torch.from_numpy(P),                # (L, 4)
            "xyz":    torch.from_numpy(xyz),              # (L, 3)
            "eot":    torch.from_numpy(d["eot"]).unsqueeze(-1),  # (L, 1)
            "length": torch.tensor(P.shape[0], dtype=torch.long),
            "name":   os.path.basename(self.files[idx]),
        }


def pad_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Right-pad sequences to max length in the batch and emit a length vector
    plus a (B, L, 1) float mask. Names are returned as a Python list."""
    B = len(batch)
    L = max(int(b["length"]) for b in batch)
    P    = torch.zeros(B, L, 4)
    xyz  = torch.zeros(B, L, 3)
    eot  = torch.zeros(B, L, 1)
    mask = torch.zeros(B, L, 1)
    lengths = torch.zeros(B, dtype=torch.long)
    names: List[str] = []
    for i, b in enumerate(batch):
        l = int(b["length"])
        P[i, :l]   = b["P"]
        xyz[i, :l] = b["xyz"]
        eot[i, :l] = b["eot"]
        mask[i, :l, 0] = 1.0
        lengths[i] = l
        names.append(b["name"])
    return {"P": P, "xyz": xyz, "eot": eot,
            "mask": mask, "lengths": lengths, "names": names}
