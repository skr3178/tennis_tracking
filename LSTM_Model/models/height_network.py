"""Height prediction sub-pipeline: LSTM^f, LSTM^b, ramp-sum, LSTM^height.

Tables 6 and 7 in the paper.
"""

import torch
import torch.nn as nn

from .lstm_blocks import FCHead, RecurrentLSTMStack, ResidualBiLSTMStack


class _DirectionalHeight(nn.Module):
    """One direction of the height predictor (LSTM^f or LSTM^b, Table 6).

    Per step input  : (ΔP_t, ε_t, h_t^d) ∈ R^6
    Per step output : Δh_t^d ∈ R, and the running accumulator h_t^d.

    Recurrence (forward):  h_0^d = 0,   h_{t+1}^d = h_t^d + Δh_t^d
    Recurrence (backward): h_N^d = 0,   h_{t-1}^d = h_t^d + Δh_t^d
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.body = RecurrentLSTMStack(in_dim=6, hidden=hidden)
        self.head = FCHead(in_dim=hidden, out_dim=1, sigmoid=False)

    def forward(self, dP: torch.Tensor, eps: torch.Tensor,
                reverse: bool = False, lengths: torch.Tensor = None):
        """If `lengths` (B,) is provided, the backward pass is anchored at
        each row's true last valid frame (lengths[b]-1) rather than at L-1,
        and updates outside the valid range are masked out.

        Forward pass is also masked beyond lengths[b], so the running h_d
        does not drift on padded frames.
        """
        B, L, _ = dP.shape
        device, dtype = dP.device, dP.dtype
        state = self.body.init_state(B, device, dtype)

        if lengths is None:
            lengths = torch.full((B,), L, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

        h_d = torch.zeros(B, 1, device=device, dtype=dtype)  # accumulator
        h_seq = [None] * L
        delta_seq = [None] * L
        zero_h = torch.zeros_like(h_d)
        zero_state_h = [torch.zeros_like(s) for s in state[0]]
        zero_state_c = [torch.zeros_like(s) for s in state[1]]

        time_idx = range(L - 1, -1, -1) if reverse else range(L)
        # Per-step active mask: forward → t < lengths[b]; backward → t < lengths[b].
        # (For backward, frames at and beyond lengths[b] are padding and must
        # not perturb the accumulator that we anchor at lengths[b]-1.)
        for t in time_idx:
            active = (t < lengths).to(dtype=dtype).view(B, 1)  # (B, 1)
            x_t = torch.cat([dP[:, t], eps[:, t], h_d], dim=-1)
            cell_out, new_state = self.body.step(x_t, state)
            delta_h = self.head(cell_out)
            h_seq[t] = h_d              # input value AT step t
            delta_seq[t] = delta_h
            # Only update h_d and the cell state within the valid range.
            h_d = torch.where(active.bool(), h_d + delta_h, h_d)
            new_h = [torch.where(active.bool(), nh, zh)
                     for nh, zh in zip(new_state[0], zero_state_h)]
            new_c = [torch.where(active.bool(), nc, zc)
                     for nc, zc in zip(new_state[1], zero_state_c)]
            state = (new_h, new_c)

        return torch.stack(h_seq, dim=1), torch.stack(delta_seq, dim=1)


class HeightNetwork(nn.Module):
    """Full height pipeline: LSTM^f + LSTM^b -> ramp sum -> LSTM^height."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.lstm_f = _DirectionalHeight(hidden=hidden)
        self.lstm_b = _DirectionalHeight(hidden=hidden)
        self.refiner_body = ResidualBiLSTMStack(in_dim=5, hidden=hidden)
        self.refiner_head = FCHead(in_dim=2 * hidden, out_dim=1, sigmoid=False)

    def forward(self, dP: torch.Tensor, eps: torch.Tensor, P: torch.Tensor,
                lengths: torch.Tensor = None):
        """dP, P: (B, L, 4); eps: (B, L, 1); optional lengths (B,) for
        variable-length batches.

        Per-row ramp weights w_t = t / (lengths[b]-1), clamped to [0, 1] so
        padded positions stay defined. The forward and backward height
        accumulators are themselves masked inside `_DirectionalHeight`.
        """
        h_f, _ = self.lstm_f(dP, eps, reverse=False, lengths=lengths)  # (B, L, 1)
        h_b, _ = self.lstm_b(dP, eps, reverse=True,  lengths=lengths)  # (B, L, 1)
        B, L, _ = dP.shape
        if lengths is None:
            lengths = torch.full((B,), L, device=dP.device, dtype=torch.long)
        else:
            lengths = lengths.to(device=dP.device, dtype=torch.long)
        denom = (lengths.to(dP.dtype) - 1.0).clamp(min=1.0).view(B, 1, 1)
        t = torch.arange(L, device=dP.device, dtype=dP.dtype).view(1, L, 1)
        w = (t / denom).clamp(0.0, 1.0)                  # (B, L, 1) per-row
        h = (1.0 - w) * h_f + w * h_b                    # Eq. 3, (B, L, 1)
        x = torch.cat([h, P], dim=-1)                    # (B, L, 5)
        h_refined = self.refiner_head(
            self.refiner_body(x, lengths=lengths))       # (B, L, 1)
        return h_refined, h
