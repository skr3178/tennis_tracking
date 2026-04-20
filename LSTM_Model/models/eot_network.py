"""LSTM^ε — End-of-Trajectory prediction network (paper Table 5)."""

import torch
import torch.nn as nn

from .lstm_blocks import FCHead, ResidualBiLSTMStack


class EoTNetwork(nn.Module):
    """Input: ΔP_t  (B, L, 4)   ->   Output: ε_t  (B, L, 1) ∈ [0, 1]."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.body = ResidualBiLSTMStack(in_dim=4, hidden=hidden)
        self.head = FCHead(in_dim=2 * hidden, out_dim=1, sigmoid=True)

    def forward(self, dP: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        return self.head(self.body(dP, lengths=lengths))
