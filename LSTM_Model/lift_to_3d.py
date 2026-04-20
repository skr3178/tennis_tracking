"""Closed-form lift from (refined height, plane points) to 3D coord (x, y, z).

Per paper §3.1.4: given the refined height h and plane-points
P = (p_g.x, p_g.z, p_v.x, p_v.y), find ray param t = h / p_v.y, then

    x = p_g.x + t * (p_v.x - p_g.x)
    y = h
    z = p_g.z * (1 - t)
"""

import torch


def lift_to_3d(h: torch.Tensor, P: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """h: (..., 1), P: (..., 4). Returns (..., 3)."""
    pg_x, pg_z, pv_x, pv_y = P.unbind(-1)
    h_s = h.squeeze(-1)
    # Paper assumes no ray is parallel to either plane, so pv_y != 0.
    # Tiny safety to avoid div-by-zero on synthetic noise.
    denom = torch.where(pv_y.abs() < eps,
                        torch.full_like(pv_y, eps),
                        pv_y)
    t = h_s / denom
    x = pg_x + t * (pv_x - pg_x)
    y = h_s
    z = pg_z * (1.0 - t)
    return torch.stack([x, y, z], dim=-1)
