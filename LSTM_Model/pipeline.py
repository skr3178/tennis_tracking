"""End-to-end forward of the four sub-networks (paper Fig. 2)."""

from typing import Dict

import torch
import torch.nn as nn

from lift_to_3d import lift_to_3d
from models.eot_network import EoTNetwork
from models.height_network import HeightNetwork
from models.refinement_network import RefinementNetwork


class WhereIsTheBall(nn.Module):
    """Input: P (B, L, 4) plane-point sequence.
    Output dict with keys 'xyz_final', 'eps', 'h_refined', 'h_combined', 'r'.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.eot = EoTNetwork(hidden=hidden)
        self.height = HeightNetwork(hidden=hidden)
        self.refine = RefinementNetwork(hidden=hidden)

    def forward(self, P: torch.Tensor,
                lengths: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """P: (B, L, 4) plane points. Optional lengths (B,) — true valid
        length per row for variable-length batches; if None, assumed L."""
        B, L, _ = P.shape
        # ΔP_t = P_{t+1} - P_t, zero-pad last step so length stays L.
        zeros_last = torch.zeros(B, 1, 4, device=P.device, dtype=P.dtype)
        dP = torch.cat([P[:, 1:] - P[:, :-1], zeros_last], dim=1)

        # For variable-length batches, kill the spike at the valid->pad
        # boundary (where dP = 0 - P[l-1]) and any junk in the padded region.
        if lengths is not None:
            lengths_l = lengths.to(device=P.device, dtype=torch.long)
            t_idx = torch.arange(L, device=P.device).view(1, L)
            valid_dP = (t_idx < (lengths_l.view(B, 1) - 1))  # need t+1 in range
            dP = dP * valid_dP.to(dP.dtype).unsqueeze(-1)

        eps = self.eot(dP, lengths=lengths)                           # (B, L, 1)
        h_refined, h_combined = self.height(dP, eps, P, lengths=lengths)
        r = lift_to_3d(h_refined, P)                                  # (B, L, 3)
        delta = self.refine(r, P, lengths=lengths)                    # (B, L, 3)
        xyz_final = r + delta

        return {
            "xyz_final":  xyz_final,
            "eps":        eps,
            "h_refined":  h_refined,
            "h_combined": h_combined,
            "r":          r,
        }
