"""LSTM^refine — 3D coordinate refinement network (paper Table 8)."""

import torch
import torch.nn as nn

from .lstm_blocks import FCHead, ResidualBiLSTMStack


class RefinementNetwork(nn.Module):
    """Input: (r_t, P_t) ∈ R^7 -> Output: (δx, δy, δz) ∈ R^3.

    The lifted 3D coordinate r_t = (x, y, z) is concatenated with the
    plane points P_t = (p_g.x, p_g.z, p_v.x, p_v.y) and fed through a
    bidirectional residual LSTM stack to produce per-step deltas.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.body = ResidualBiLSTMStack(in_dim=7, hidden=hidden)
        self.head = FCHead(in_dim=2 * hidden, out_dim=3, sigmoid=False)

    def forward(self, r: torch.Tensor, P: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        x = torch.cat([r, P], dim=-1)                    # (B, L, 7)
        return self.head(self.body(x, lengths=lengths))  # (B, L, 3)
