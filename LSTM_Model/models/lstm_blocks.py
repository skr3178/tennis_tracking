"""Reusable building blocks for the four sub-networks (Tables 5-8)."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FCHead(nn.Module):
    """Three LeakyReLU(0.01) hidden FC layers + final linear (optional sigmoid).

    Matches the FC.0..FC.3 stack at the bottom of Tables 5/6/7/8:
    in_dim -> 32 -> 32 -> 32 -> out_dim.
    """

    def __init__(self, in_dim: int, hidden: int = 32,
                 out_dim: int = 1, sigmoid: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.01),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.01),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.01),
            nn.Linear(hidden, out_dim),
        )
        self.sigmoid = sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.sigmoid:
            y = torch.sigmoid(y)
        return y


class ResidualBiLSTMStack(nn.Module):
    """3 stacked BiLSTMs with a ResNet-style shortcut: out_2 + out_0.

    Matches Tables 5 / 7 / 8.  Output shape: (B, L, 2*hidden) = (B, L, 128).
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        bi_dim = 2 * hidden
        self.lstm0 = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.lstm1 = nn.LSTM(bi_dim, hidden, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(bi_dim, hidden, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        """If `lengths` (B,) is provided, the bidirectional LSTMs use packed
        sequences so the backward direction does NOT see padded frames.
        Outputs at padded positions are zero-filled."""
        if lengths is None:
            out0, _ = self.lstm0(x)
            out1, _ = self.lstm1(out0)
            out2, _ = self.lstm2(out1)
            return out2 + out0

        L = x.size(1)
        lengths_cpu = lengths.detach().cpu().to(torch.long)
        packed0 = pack_padded_sequence(x, lengths_cpu, batch_first=True,
                                        enforce_sorted=False)
        out0_p, _ = self.lstm0(packed0)
        out1_p, _ = self.lstm1(out0_p)
        out2_p, _ = self.lstm2(out1_p)
        out0, _ = pad_packed_sequence(out0_p, batch_first=True, total_length=L)
        out2, _ = pad_packed_sequence(out2_p, batch_first=True, total_length=L)
        return out2 + out0


class RecurrentLSTMStack(nn.Module):
    """3 stacked unidirectional LSTMCells, exposed step-by-step.

    Used by LSTM^f / LSTM^b (Table 6) where the input at step t depends on
    the running height accumulator h_t^d, so we cannot vectorize over time.
    Output per step: (B, hidden) — passed to FCHead afterwards.
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.cell0 = nn.LSTMCell(in_dim, hidden)
        self.cell1 = nn.LSTMCell(hidden, hidden)
        self.cell2 = nn.LSTMCell(hidden, hidden)

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        z = lambda: torch.zeros(batch_size, self.hidden, device=device, dtype=dtype)
        return [z(), z(), z()], [z(), z(), z()]

    def step(self, x_t: torch.Tensor,
             state: Tuple[List[torch.Tensor], List[torch.Tensor]]
             ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        h, c = state
        h0, c0 = self.cell0(x_t, (h[0], c[0]))
        h1, c1 = self.cell1(h0,  (h[1], c[1]))
        h2, c2 = self.cell2(h1,  (h[2], c[2]))
        return h2, ([h0, h1, h2], [c0, c1, c2])
