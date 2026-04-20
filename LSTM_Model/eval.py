"""Evaluation: NRMSE on distance and height, matching paper §4.1 / Table 12.

NRMSE definition (standard):
    NRMSE = RMSE(pred, gt) / (max(gt) - min(gt))

For "distance NRMSE" we collect the per-frame Euclidean error e_t =
||xyz_pred_t - xyz_gt_t|| across all valid frames, then
    RMSE = sqrt(mean(e_t^2))
    range = max(||xyz_gt||) over the eval set
    NRMSE = RMSE / range

For "height NRMSE" we use only the y-axis difference and y-range.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from config import EvalConfig
from data.dataset import BallTrajectoryDataset, pad_collate
from pipeline import WhereIsTheBall


def nrmse_distance_height(net: WhereIsTheBall,
                           loader: Iterable,
                           device: torch.device) -> Dict[str, float]:
    """Walks `loader` once and returns {'nrmse_distance', 'nrmse_height',
    'mean_dist_err_m', 'mean_height_err_m'}."""
    net.eval()
    sq_dist_sum = 0.0
    sq_h_sum = 0.0
    n_valid = 0
    abs_dist_sum = 0.0
    abs_h_sum = 0.0
    gt_dist_max = 0.0
    gt_h_min = float("inf"); gt_h_max = float("-inf")
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            out = net(batch["P"], lengths=batch["lengths"])
            mask = batch["mask"]                               # (B, L, 1)
            mb = mask.bool().squeeze(-1)
            d = (out["xyz_final"] - batch["xyz"])              # (B, L, 3)
            d_norm = d.pow(2).sum(-1).sqrt()                   # (B, L)
            h_err = d[..., 1].abs()                            # (B, L)
            sq_dist_sum += float((d_norm.pow(2) * mb).sum())
            sq_h_sum    += float((h_err.pow(2) * mb).sum())
            abs_dist_sum += float((d_norm * mb).sum())
            abs_h_sum    += float((h_err * mb).sum())
            n_valid += int(mb.sum())
            gt_dist_max = max(gt_dist_max,
                              float((batch["xyz"].pow(2).sum(-1).sqrt() * mb).max()))
            ys = batch["xyz"][..., 1][mb]
            if ys.numel():
                gt_h_min = min(gt_h_min, float(ys.min()))
                gt_h_max = max(gt_h_max, float(ys.max()))
    n_valid = max(1, n_valid)
    rmse_dist = (sq_dist_sum / n_valid) ** 0.5
    rmse_h    = (sq_h_sum    / n_valid) ** 0.5
    h_range = max(gt_h_max - gt_h_min, 1e-6)
    return {
        "nrmse_distance":   rmse_dist / max(gt_dist_max, 1e-6),
        "nrmse_height":     rmse_h / h_range,
        "mean_dist_err_m":  abs_dist_sum / n_valid,
        "mean_height_err_m": abs_h_sum / n_valid,
    }


def evaluate(cfg: EvalConfig) -> Dict[str, float]:
    device = torch.device(cfg.device if torch.cuda.is_available()
                           and cfg.device.startswith("cuda") else "cpu")
    ds = BallTrajectoryDataset(
        root=cfg.data_root, split_subdir=cfg.split_subdir, split=cfg.split,
        train_frac=cfg.train_frac, val_frac=cfg.val_frac, test_frac=cfg.test_frac,
        seed=cfg.seed, uv_noise_sigma_px=0.0,
    )
    loader = DataLoader(ds, batch_size=max(1, min(64, len(ds))), shuffle=False,
                        collate_fn=pad_collate)

    net = WhereIsTheBall().to(device)
    if not os.path.isfile(cfg.ckpt):
        raise FileNotFoundError(cfg.ckpt)
    state = torch.load(cfg.ckpt, map_location=device)
    net.load_state_dict(state["model_state"])
    print(f"[eval] loaded {cfg.ckpt}  epoch={state.get('epoch', '?')}  "
          f"|{cfg.split}|={len(ds)}")

    metrics = nrmse_distance_height(net, loader, device)
    print(f"[eval] {cfg.split}  "
          f"NRMSE_distance = {metrics['nrmse_distance']:.4%}  "
          f"NRMSE_height = {metrics['nrmse_height']:.4%}  "
          f"mean_dist_err = {metrics['mean_dist_err_m']:.4f} m  "
          f"mean_height_err = {metrics['mean_height_err_m']:.4f} m")
    return metrics


def _parse_args() -> EvalConfig:
    cfg = EvalConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=cfg.ckpt)
    p.add_argument("--data_root", default=cfg.data_root)
    p.add_argument("--split_subdir", default=cfg.split_subdir,
                   choices=["synthetic", "real", "ours_game_1000", "ours_game_1000_trimmed", "ours_game_5000_trimmed", "combined"])
    p.add_argument("--split", default=cfg.split, choices=["train", "val", "test"])
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--seed", type=int, default=cfg.seed)
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    evaluate(_parse_args())
