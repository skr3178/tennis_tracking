"""Training entry-point. Paper §D recipe:
    - Adam, lr 1e-3
    - batch size 256
    - 1,400 epochs
    - Gaussian noise augmentation on (u, v) before parameterization
    - loss weights (λ_ε, λ_3D, λ_B) = (10, 1, 10)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from data.dataset import BallTrajectoryDataset, pad_collate
from eval import nrmse_distance_height
from losses import LossConfig, total_loss
from pipeline import WhereIsTheBall


def _make_dataset(cfg: TrainConfig, split: str) -> BallTrajectoryDataset:
    return BallTrajectoryDataset(
        root=cfg.data_root, split_subdir=cfg.split_subdir, split=split,
        train_frac=cfg.train_frac, val_frac=cfg.val_frac, test_frac=cfg.test_frac,
        seed=cfg.seed,
        uv_noise_sigma_px=cfg.uv_noise_sigma_px if split == "train" else 0.0,
        clamp_gt_y_nonneg=cfg.clamp_gt_y_nonneg,
    )


def _move(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def train(cfg: TrainConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available()
                           and cfg.device.startswith("cuda") else "cpu")
    print(f"[train] device={device}  data_root={cfg.data_root}  "
          f"split_subdir={cfg.split_subdir}")

    # Datasets + loaders. Cap batch_size by train-set size for tiny splits.
    train_ds = _make_dataset(cfg, "train")
    val_ds   = _make_dataset(cfg, "val")
    bs_train = max(1, min(cfg.batch_size, len(train_ds)))
    bs_val   = max(1, min(cfg.batch_size, len(val_ds)))
    train_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=pad_collate)
    val_loader   = DataLoader(val_ds, batch_size=bs_val, shuffle=False,
                              num_workers=cfg.num_workers, collate_fn=pad_collate)
    print(f"[train] |train|={len(train_ds)}  |val|={len(val_ds)}  "
          f"bs_train={bs_train}  bs_val={bs_val}")

    net = WhereIsTheBall(hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    loss_cfg = LossConfig(lambda_eps=cfg.lambda_eps, lambda_3d=cfg.lambda_3d,
                          lambda_b=cfg.lambda_b,
                          eot_gamma=None if cfg.eot_gamma < 0 else cfg.eot_gamma)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_metric = math.inf
    log_path = os.path.join(cfg.ckpt_dir, "train_log.jsonl")
    log_f = open(log_path, "a")

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        epoch_sums = {"total": 0.0, "L_eps": 0.0, "L_3D": 0.0, "L_B": 0.0}
        n_batches = 0
        iterator = train_loader
        if cfg.use_tqdm:
            iterator = tqdm(train_loader,
                            desc=f"ep {epoch:>4d}/{cfg.epochs}",
                            leave=False, ncols=110)
        for batch in iterator:
            batch = _move(batch, device)
            out = net(batch["P"], lengths=batch["lengths"])
            losses = total_loss(out, batch, loss_cfg)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            for k in epoch_sums:
                epoch_sums[k] += float(losses[k].detach().cpu())
            n_batches += 1
            if cfg.use_tqdm:
                iterator.set_postfix({
                    "L":     f"{epoch_sums['total']/n_batches:.4f}",
                    "L_eps": f"{epoch_sums['L_eps']/n_batches:.4f}",
                    "L_3D":  f"{epoch_sums['L_3D']/n_batches:.4f}",
                    "L_B":   f"{epoch_sums['L_B']/n_batches:.4f}",
                })

        for k in epoch_sums:
            epoch_sums[k] /= max(1, n_batches)

        do_log = (epoch % cfg.log_every == 0) or epoch == 1 or epoch == cfg.epochs
        do_val = (epoch % cfg.val_every == 0) or epoch == cfg.epochs

        val_metrics = None
        if do_val:
            val_metrics = nrmse_distance_height(net, val_loader, device)

        if do_log:
            elapsed = time.time() - t0
            msg = (f"[ep {epoch:5d}/{cfg.epochs}] "
                   f"L={epoch_sums['total']:.4f}  "
                   f"L_eps={epoch_sums['L_eps']:.4f}  "
                   f"L_3D={epoch_sums['L_3D']:.4f}  "
                   f"L_B={epoch_sums['L_B']:.4f}  "
                   f"({elapsed:.0f}s)")
            if val_metrics is not None:
                msg += (f"  | val NRMSE_dist={val_metrics['nrmse_distance']:.4f}"
                        f"  NRMSE_h={val_metrics['nrmse_height']:.4f}")
            print(msg)

        rec = {"epoch": epoch, **epoch_sums}
        if val_metrics is not None:
            rec.update({f"val_{k}": v for k, v in val_metrics.items()})
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()

        if val_metrics is not None:
            if val_metrics["nrmse_distance"] < best_metric:
                best_metric = val_metrics["nrmse_distance"]
                ckpt_path = os.path.join(cfg.ckpt_dir, "best.pt")
                torch.save({"epoch": epoch,
                            "model_state": net.state_dict(),
                            "val_metrics": val_metrics,
                            "cfg": cfg.__dict__}, ckpt_path)

    torch.save({"epoch": cfg.epochs,
                "model_state": net.state_dict(),
                "cfg": cfg.__dict__},
                os.path.join(cfg.ckpt_dir, "last.pt"))
    log_f.close()
    print(f"[train] done. best val NRMSE_distance = {best_metric:.4f}")


def _parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=cfg.data_root)
    p.add_argument("--split_subdir", default=cfg.split_subdir,
                   choices=["synthetic", "real", "ours_game_1000", "ours_game_1000_trimmed", "ours_game_5000_trimmed", "combined"])
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--ckpt_dir", default=cfg.ckpt_dir)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--uv_noise_sigma_px", type=float, default=cfg.uv_noise_sigma_px)
    p.add_argument("--smoketest", action="store_true",
                   help="5 epochs, batch_size=4, log every epoch")
    p.add_argument("--no_clamp_y", action="store_true",
                   help="Do not clamp GT y >= 0 (let L_B see raw bounce slack)")
    args = p.parse_args()
    for k, v in vars(args).items():
        if k == "smoketest" or k == "no_clamp_y":
            continue
        setattr(cfg, k, v)
    if args.smoketest:
        cfg.epochs = 5
        cfg.batch_size = 4
        cfg.log_every = 1
        cfg.val_every = 1
    if args.no_clamp_y:
        cfg.clamp_gt_y_nonneg = False
    return cfg


if __name__ == "__main__":
    train(_parse_args())
