# Implement "Where Is The Ball" LSTM Pipeline (Tennis)

## Context

Reproduce the best-performing configuration of *Where Is The Ball: 3D Ball Trajectory Estimation From 2D Monocular Tracking* (Ponglertnapakorn & Suwajanakorn, arXiv 2506.05763, 2025) for the **synthetic Tennis** setting only. The paper proposes an LSTM-based pipeline with a novel canonical 3D representation (plane-points) that converts a 2D ball track `(u_t, v_t)` plus camera params into a 3D trajectory `(x_t, y_t, z_t)`.

Per ablation tables 13–15, the winning recipe is:
- **Input/output parameterization**: `P = (p_ground, p_vertical) ∈ R⁴` → **height** output (refined to xyz). Discard pixel / pixel+E / azimuth-elevation alternatives.
- **All four LSTM components active**: EoT (ε), forward+backward height (f,b), bidirectional height refiner, 3D refinement.
- **All three losses active**: `L_ε` (weighted BCE), `L_3D` (L2), `L_B` (below-ground penalty).

User confirmed: implement *only* this winning path — no toggles for losing variants.

All artifacts live in `/media/skr/storage/ten_bad/LSTM_Model/`. Use the project venv `/media/skr/storage/ten_bad/.venv` and `uv pip install` per project rules.

---

## File layout

```
LSTM_Model/
├── PLAN.md                      # this plan
├── config.py                    # all hyperparameters + paths
├── data/
│   ├── dataset.py               # SyntheticTennisDataset (paper-spec npz)
│   ├── parameterization.py      # 2D pixel ↔ ray ↔ (p_ground, p_vertical)
│   └── synth_smoketest.py       # tiny PhysX-free generator for pipeline smoke-test
├── models/
│   ├── lstm_blocks.py           # ResidualBiLSTMStack + FCHead (Tables 5/7/8)
│   ├── eot_network.py           # LSTM^ε  (Table 5)
│   ├── height_network.py        # LSTM^f, LSTM^b, LSTM^height (Tables 6, 7)
│   └── refinement_network.py    # LSTM^refine (Table 8)
├── pipeline.py                  # end-to-end forward (chains all 4 nets)
├── losses.py                    # L_ε, L_3D, L_B + total weighted loss
├── lift_to_3d.py                # height + plane-points → (x,y,z) closed-form
├── train.py                     # 1,400-epoch training loop, Adam lr=1e-3, bs=256
├── eval.py                      # NRMSE distance + height on test split
├── checkpoints/                 # saved weights
└── outputs/                     # predicted trajectories (npz)
```

---

## Component design

### 1. Plane-points parameterization (`data/parameterization.py`)

Per paper §3.1.1 / Eqs. 1–2. Given pixel `(u,v)`, intrinsics `(f, p_x, p_y)`, and extrinsic `E ∈ SE(3)`:
- Camera center `c = ψ(E⁻¹ [0,0,0,1]ᵀ)`
- Ray direction `d = ψ(E⁻¹ [u−p_x, v−p_y, f, 0]ᵀ)`
- `p_ground = ray ∩ {y=0}` → drop y → `(x, z)`
- `p_vertical = ray ∩ {z=0}` → drop z → `(x, y)`
- Output `P_t ∈ R⁴ = (p_ground_x, p_ground_z, p_vertical_x, p_vertical_y)`

Lift back (`lift_to_3d.py`): given height `h` and `(p_g, p_v)`, ray param `t = h / p_v_y`, then
`x = p_g_x + t·(p_v_x − p_g_x)`, `y = h`, `z = p_g_z·(1 − t)`.

### 2. Reusable LSTM blocks (`models/lstm_blocks.py`)

`ResidualBiLSTMStack(in_dim, hidden=64, num_layers=3)` matching Tables 5/7/8:
- 3 BiLSTMs (hidden=64 each → output 128) with residual: `out_2 = BiLSTM_2(out_1) + out_0`.
- "Concat" in tables = passthrough of the residual sum (already 128-dim).
- `FCHead`: 3× `Linear(→32) + LeakyReLU(0.01)` then `Linear(→out_dim)` (sigmoid only on EoT head).

For unidirectional `LSTM^f` / `LSTM^b` (Table 6): `ResidualLSTMStack(bidir=False)` — same shape, hidden=64, output=64.

### 3. EoT network (`models/eot_network.py`) — LSTM^ε

- Input: `ΔP_t = P_{t+1} − P_t` (zero-padded last step). Shape `B×L×4`.
- `ResidualBiLSTMStack(in_dim=4)` + `FCHead(out_dim=1, sigmoid)` → `ε_t ∈ [0,1]`.

### 4. Height prediction (`models/height_network.py`)

Two unidirectional LSTMs accumulate height *deltas*:
- `LSTM^f`: input `(ΔP_t, ε_t, h_t^f)` → `Δh_t^f`. Step-wise loop: `h^f_t = h^f_{t−1} + Δh^f_{t−1}`, `h^f_0 = 0`.
- `LSTM^b`: same arch, backward, `h^b_N = 0`.
- Combine via Eq. 3 ramp sum: `w_t = (t−1)/(N−1)`, `h_t = (1−w_t) h^f_t + w_t h^b_t`.

Then `LSTM^height` (bidir): input `(h_t, P_t) ∈ R⁵` → `h_t^refined`. `ResidualBiLSTMStack(in_dim=5)` + `FCHead(out_dim=1)`.

### 5. Refinement network (`models/refinement_network.py`) — LSTM^refine

- `r_t = lift_to_3d(h_t^refined, P_t)` → `(x_t, y_t, z_t)`.
- Concat with `P_t` → `B×L×7`.
- `ResidualBiLSTMStack(in_dim=7)` + `FCHead(out_dim=3)` → `(δx, δy, δz)`.
- Final: `(x,y,z)_final = r_t + (δx, δy, δz)`.

### 6. Pipeline (`pipeline.py`)

```python
class WhereIsTheBall(nn.Module):
    def forward(self, P):
        dP   = torch.cat([P[:,1:]-P[:,:-1], zeros(B,1,4)], 1)
        eps  = self.eot(dP)
        h_f  = self.lstm_f(dP, eps)             # recurrent
        h_b  = self.lstm_b(dP, eps)             # recurrent backward
        h    = ramp_sum(h_f, h_b)
        h_r  = self.lstm_height(torch.cat([h, P], -1))
        r    = lift_to_3d(h_r, P)
        delta= self.lstm_refine(torch.cat([r, P], -1))
        return r + delta, eps
```

### 7. Losses (`losses.py`)

- `L_ε`: weighted BCE per Eq. 4 with `γ` balancing positives.
- `L_3D`: mean L2 between `xyz_final` and GT (Eq. 5).
- `L_B`: mean of `y²` over predictions where `y < 0` (Eq. 6).
- Total: `(λ_ε, λ_3D, λ_B) = (10, 1, 10)` per §D.

### 8. Training (`train.py`)

- Adam, lr `1e-3`, batch size `256`, **1,400 epochs** (paper).
- BPTT on full sequences. Add Gaussian noise to `(u,v)` *before* parameterization on each batch (paper recipe, not an ablation).
- Sequences pad-collated; losses masked to valid steps.
- Save best-by-val-NRMSE checkpoint to `checkpoints/best.pt`.

### 9. Evaluation (`eval.py`)

- NRMSE distance and height on the test split (Table 12 metric).

---

## Dataset format the loader expects

`SyntheticTennisDataset` consumes `.npz` files (per §C: 5000 train / 1500 val / 500 test, length 64–822, ~3 strokes each):

| key | shape | dtype | meaning |
|---|---|---|---|
| `uv` | `(L, 2)` | float32 | 2D ball track in pixels |
| `xyz` | `(L, 3)` | float32 | GT 3D position, `y=0` ground, `y+` up |
| `eot` | `(L,)` | uint8 | 1 just before a hit/force |
| `intrinsics` | `(3,)` | float32 | `(f, p_x, p_y)` |
| `extrinsic` | `(4, 4)` | float32 | `E ∈ SE(3)`, world→camera |

Layout: `data_root/{train,val,test}/seq_{idx:05d}.npz`. Path in `config.py`.

`data/synth_smoketest.py` generates ~50 toy sequences in this exact format for end-to-end smoke testing.

---

## Verification

```bash
cd /media/skr/storage/ten_bad/LSTM_Model
/media/skr/storage/ten_bad/.venv/bin/python data/synth_smoketest.py
/media/skr/storage/ten_bad/.venv/bin/python train.py --epochs 5 --smoketest
/media/skr/storage/ten_bad/.venv/bin/python eval.py --ckpt checkpoints/best.pt
```

Pass criteria:
1. Forward pass shapes match Tables 5–8 (model `__main__` assertions).
2. `lift_to_3d` round-trips a known 3D point with ≤1e-4 error.
3. After 5 smoke epochs, loss decreases and `eval.py` reports finite NRMSE.

Real-data run (5000-seq dataset):

```bash
/media/skr/storage/ten_bad/.venv/bin/python train.py
```

Target per Table 12: distance NRMSE ≈ 0.15% (≈8 cm) on Synthetic Tennis test.
