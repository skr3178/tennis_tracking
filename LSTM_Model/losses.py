"""Three loss components from paper §3.2 (Eqs. 4-7).

All losses are masked by the per-step validity mask so padded frames in
variable-length batches do not contribute. Total loss:

    L = λ_ε · L_ε  +  λ_3D · L_3D  +  λ_B · L_B
    (10,    1,    10)  per paper §D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


_EPS_LOG = 1e-6


def weighted_bce_eot(eps_pred: torch.Tensor,
                      eot_gt: torch.Tensor,
                      mask: torch.Tensor,
                      gamma: Optional[float] = None) -> torch.Tensor:
    """Eq. 4: weighted BCE on per-step EoT prediction.

    eps_pred: (B, L, 1) in (0, 1)
    eot_gt:   (B, L, 1) in {0, 1}
    mask:     (B, L, 1) in {0, 1}
    gamma:    positive-class weight; if None, computed per-batch as
              (#neg / #pos), clamped to [1, 1e3] to keep gradients sane.
    """
    eps_pred = eps_pred.clamp(_EPS_LOG, 1.0 - _EPS_LOG)
    pos = (eot_gt * mask).sum().clamp(min=1.0)
    neg = ((1.0 - eot_gt) * mask).sum().clamp(min=1.0)
    g = (neg / pos).clamp(1.0, 1e3) if gamma is None else torch.tensor(
        float(gamma), device=eps_pred.device, dtype=eps_pred.dtype)
    pos_term = g * eot_gt * eps_pred.log()
    neg_term = (1.0 - eot_gt) * (1.0 - eps_pred).log()
    per_step = -(pos_term + neg_term) * mask
    return per_step.sum() / mask.sum().clamp(min=1.0)


def l2_3d(xyz_pred: torch.Tensor, xyz_gt: torch.Tensor,
           mask: torch.Tensor) -> torch.Tensor:
    """Eq. 5: mean squared error on 3D coords (per masked frame, per axis)."""
    sq = (xyz_pred - xyz_gt).pow(2) * mask
    n_terms = mask.sum().clamp(min=1.0) * 3.0
    return sq.sum() / n_terms


def below_ground(xyz_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Eq. 6: penalty on negative y predictions, mean over masked frames."""
    y = xyz_pred[..., 1:2]
    pen = F.relu(-y).pow(2) * mask
    return pen.sum() / mask.sum().clamp(min=1.0)


@dataclass
class LossConfig:
    lambda_eps: float = 10.0
    lambda_3d:  float = 1.0
    lambda_b:   float = 10.0
    eot_gamma: Optional[float] = None     # None => auto from pos/neg


def total_loss(out: Dict[str, torch.Tensor],
               batch: Dict[str, torch.Tensor],
               cfg: LossConfig) -> Dict[str, torch.Tensor]:
    """Returns {'total', 'L_eps', 'L_3D', 'L_B'}; differentiable."""
    mask = batch["mask"]
    L_eps = weighted_bce_eot(out["eps"], batch["eot"], mask, cfg.eot_gamma)
    L_3D  = l2_3d(out["xyz_final"], batch["xyz"], mask)
    L_B   = below_ground(out["xyz_final"], mask)
    total = cfg.lambda_eps * L_eps + cfg.lambda_3d * L_3D + cfg.lambda_b * L_B
    return {"total": total, "L_eps": L_eps, "L_3D": L_3D, "L_B": L_B}
