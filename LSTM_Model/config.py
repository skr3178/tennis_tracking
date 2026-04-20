"""Hyperparameters and paths for the Where-Is-The-Ball pipeline.

All numbers here come from paper §D unless flagged otherwise.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainConfig:
    # --- Data ---
    data_root: str = "/media/skr/storage/ten_bad/paper_npz_rev1"
    split_subdir: str = "synthetic"          # "synthetic" or "real"
    # rev1 has 52 synthetic / 21 real sequences total. Paper splits 5000/1500/500.
    # Use fractions so the same code works on the full dataset later.
    train_frac: float = 0.72
    val_frac:   float = 0.18
    test_frac:  float = 0.10
    seed: int = 0

    # --- Augmentation (paper §D recipe, NOT an ablation) ---
    uv_noise_sigma_px: float = 1.0           # Gaussian noise added to (u, v)

    # --- Model ---
    hidden: int = 64

    # --- Optimizer (paper §D) ---
    optimizer: str = "adam"
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 1400
    grad_clip: float = 5.0                   # not in paper; safety for BPTT

    # --- Loss weights (paper §D) ---
    lambda_eps: float = 10.0
    lambda_3d:  float = 1.0
    lambda_b:   float = 10.0
    eot_gamma: float = -1.0                  # <0 -> auto from pos/neg ratio per batch
    clamp_gt_y_nonneg: bool = True           # see CHANGELOG; rev1 has small y<0 values

    # --- Logging / IO ---
    ckpt_dir: str = "checkpoints"
    log_every: int = 1                       # print epoch losses every N epochs
    val_every: int = 10
    use_tqdm: bool = True                    # per-batch progress bar with running loss
    device: str = "cuda"                     # falls back to cpu in train.py if unavailable
    num_workers: int = 0


@dataclass
class EvalConfig:
    data_root: str = "/media/skr/storage/ten_bad/paper_npz_rev1"
    split_subdir: str = "synthetic"
    split: str = "test"                      # "train", "val", or "test"
    ckpt: str = "checkpoints/best.pt"
    device: str = "cuda"
    seed: int = 0
    train_frac: float = 0.72
    val_frac:   float = 0.18
    test_frac:  float = 0.10
