# CHANGELOG

All notable changes to the LSTM_Model implementation. Newest at top.

---

## 2026-04-18 — Training, evaluation, dataset, losses, parameterization

Added the full data + training + evaluation pipeline per paper §C / §D. Per-epoch
loss components and per-batch tqdm progress are streamed to stdout so the user
sees `L_eps`, `L_3D`, `L_B`, total `L`, plus validation NRMSE, on every epoch.

### New files

- `config.py` — `TrainConfig` and `EvalConfig` dataclasses with paper §D
  defaults: Adam lr=1e-3, batch=256, epochs=1400, λ=(10,1,10), uv noise
  σ=1 px. Adds two non-paper safety knobs: `grad_clip=5.0` and
  `clamp_gt_y_nonneg=True` (see CHANGELOG note about negative GT y values).
- `data/__init__.py` — package marker.
- `data/parameterization.py` — vectorized `pixel_to_plane_points(uv,
  intrinsics, extrinsic, convention)` per Eqs. 1-2. Handles both `opengl`
  (rev1's convention) and `opencv` ray sign conventions, picks the correct
  one from `camera.json` automatically.
- `data/dataset.py` — `BallTrajectoryDataset` reads `seq_*.npz` from a
  split dir, splits 72/18/10 train/val/test by seed, applies Gaussian noise
  to `(u, v)` per the paper recipe (training-only), parameterizes on the
  fly, optionally clamps GT y ≥ 0. `pad_collate` produces `(P, xyz, eot,
  mask, lengths, names)` so the mask-aware pipeline gets `lengths` directly.
- `losses.py` — `weighted_bce_eot` (Eq. 4, with auto-γ from the per-batch
  pos/neg ratio), `l2_3d` (Eq. 5), `below_ground` (Eq. 6), `total_loss`
  combining them with `LossConfig` weights. All masked.
- `train.py` — Adam, optional `--smoketest` (5 ep, bs=4), tqdm per-batch
  progress with running `(L, L_eps, L_3D, L_B)`, per-epoch summary, val
  NRMSE every `val_every` epochs, best checkpoint saved by val
  NRMSE_distance. Logs JSONL to `<ckpt_dir>/train_log.jsonl`.
- `eval.py` — `nrmse_distance_height` walks a loader once and returns
  `nrmse_distance`, `nrmse_height`, plus mean L1 errors in metres. Same
  function is reused inside `train.py` for validation.

### Verification (rev1 synthetic, smoketest, CPU)

```
[train] |train|=37  |val|=9  bs_train=4  bs_val=4
[ep 1/5] L=21.14  L_eps=1.21  L_3D=8.99  L_B=0.00  | val NRMSE_dist=0.097
[ep 5/5] L=14.46  L_eps=1.21  L_3D=2.32  L_B=0.00  | val NRMSE_dist=0.064
[eval] test  NRMSE_distance=7.9%  NRMSE_height=21.8%  mean_dist_err=2.09 m
```

Loss decreases monotonically; gradients flow through the mask-aware paths;
checkpoint loads cleanly into `eval.py`. NRMSE numbers are far from the
paper's 0.15% target — expected, since this is 5 epochs on 37 sequences vs
1400 epochs on 5000 sequences.

### Known follow-ups for the real run

- Run the full 1400-epoch schedule once a real-sized training set lands (the
  rev1 synthetic split has 37 train sequences total).
- Tune `eot_gamma` if the auto pos/neg ratio gives unstable BCE on small
  batches.
- Decide whether to keep `clamp_gt_y_nonneg=True` or feed the raw bounce
  slack into `L_B` as soft penalty (use `--no_clamp_y` to compare).

---

## 2026-04-18 — Mask-aware variable-length batches

### Problem

Running rev1 sample data (`/media/skr/storage/ten_bad/paper_npz_rev1`) revealed
that the same input sequence produced different predictions depending on what
it was padded against in a batch. With `B=2`, a 69-frame sequence padded
alongside a 133-frame sequence diverged from its solo prediction by **up to
20 mm at the end of the trajectory** (10 µm at the start). This means:

- Identical strokes get different gradients depending on batch composition.
- Reproducibility breaks across different shuffles.
- Trained model becomes sensitive to padding length at inference.

### Root causes

Three defects across `pipeline.py`, `models/height_network.py`,
`models/lstm_blocks.py`:

1. **`LSTM^b` anchored at frame `L−1` of the padded tensor**, not at each
   row's true last valid frame. Backward height accumulator started from junk
   and propagated wrong values through the valid region.
2. **Ramp-sum weights computed with global `L`**: `w_t = t / (L−1)`. For a
   row with valid length `l < L`, the true last valid frame received weight
   `(l−1)/(L−1) < 1` instead of `1`, mis-mixing forward and backward
   accumulators at the end of every short sequence.
3. **`dP` boundary spike**: at the valid→pad transition, `dP[l−1] = 0 −
   P[l−1]` injected a large fake delta into the EoT and Height networks.
4. **Bidirectional `nn.LSTM`s in EoT/Refinement/Height-refiner** processed
   the entire padded tensor; their backward direction read padded inputs and
   contaminated outputs at valid frames.

### Fix

Threaded a new optional `lengths: torch.LongTensor (B,)` argument through:

- `WhereIsTheBall.forward(P, lengths=None)` — `pipeline.py:25`
- `EoTNetwork.forward(dP, lengths=None)` — `models/eot_network.py:17`
- `HeightNetwork.forward(dP, eps, P, lengths=None)` — `models/height_network.py:61`
- `_DirectionalHeight.forward(dP, eps, reverse, lengths=None)` — `models/height_network.py:27`
- `RefinementNetwork.forward(r, P, lengths=None)` — `models/refinement_network.py:22`
- `ResidualBiLSTMStack.forward(x, lengths=None)` — `models/lstm_blocks.py:47`

Per-component fixes:

- **`pipeline.py`**: zero `dP` at and beyond `lengths[b]−1` (kills boundary
  spike).
- **`_DirectionalHeight`**: per-step `active = (t < lengths[b])` mask;
  when inactive, hold `h_d` constant and zero the cell state. The backward
  pass therefore stays anchored at zero through padding and starts
  accumulating only from the true last valid frame.
- **`HeightNetwork`**: per-row ramp weights `w = t / (lengths[b]−1)` clamped
  to `[0, 1]`, replacing the global-`L` formula.
- **`ResidualBiLSTMStack`**: when `lengths` is provided, use
  `pack_padded_sequence` / `pad_packed_sequence` (with `enforce_sorted=False`)
  so the bidirectional `nn.LSTM`s never read padded inputs in either
  direction. The residual skip works on padded outputs (zeros at padded
  positions, same on both branches → fine).

Default behavior (no `lengths` passed) is unchanged — old call sites and the
shape-test fixtures still work.

### Verification

| Output | Before fix (max alone vs padded diff) | After fix |
|---|---|---|
| `eps`         | 1.26e-3   | **6e-8**   |
| `h_refined`   | 8.35e-3   | **1.5e-8** |
| `h_combined`  | (large)   | **4.8e-7** |
| `r`           | (large)   | **1.9e-6** |
| `xyz_final`   | 2.05e-2 m | **1.9e-6 m** |

All deltas are now at float32 numerical noise. Added permanent regression
test `test_mask_equivalence()` in `test_shapes.py:152` that fails if any
output diverges by more than 1e-4 between the alone and padded runs.

### Files touched

- `pipeline.py`
- `models/lstm_blocks.py`
- `models/eot_network.py`
- `models/height_network.py`
- `models/refinement_network.py`
- `test_shapes.py` (added `test_mask_equivalence`)

### Backward compatibility

All four sub-network forwards accept `lengths=None` and degrade to the
original behavior. `test_shapes.py` (which never passes `lengths`) continues
to pass with no edits to the existing tests.

### Known issues NOT addressed in this change

- `xyz.y` in rev1 data still has small negative values (synthetic min
  −0.236 m, real min −0.198 m). The `L_B` (below-ground) loss in the planned
  `losses.py` will fight against matching these. Decide at loss-implementation
  time whether to clamp GT y to 0.
- EoT positive density on synthetic is ~12% (vs ~3% on real). The weighted-BCE
  γ should be set per-batch from `pos_count / neg_count`, not a fixed value.

---

## 2026-04-18 — Initial pipeline implementation

Built the four sub-networks per paper Tables 5–8:

- `models/lstm_blocks.py`: `FCHead`, `ResidualBiLSTMStack` (Tables 5/7/8),
  `RecurrentLSTMStack` (Table 6, used per-step).
- `models/eot_network.py`: `EoTNetwork` (Table 5).
- `models/height_network.py`: `_DirectionalHeight` ×2 (Table 6) + `HeightNetwork`
  refiner (Table 7), with Eq. 3 ramp sum.
- `models/refinement_network.py`: `RefinementNetwork` (Table 8).
- `lift_to_3d.py`: closed-form `(h, P) → (x, y, z)` with `t = h / p_v.y`.
- `pipeline.py`: `WhereIsTheBall` chains EoT → Height → lift → Refine.
- `test_shapes.py`: shape tests for all blocks/sub-networks/full pipeline at
  `(B, L) ∈ {(2,8), (4,64), (1,122), (3,256)}` plus an end-to-end backward
  smoke test.

Total params: **902,855** (~3.44 MB).

All shape tests pass; all 136 parameters receive non-zero gradients in the
backward smoke test.
