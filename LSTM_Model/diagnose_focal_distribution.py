#!/usr/bin/env python3
"""Standalone focal/depth distribution check.

Question: do PnP's inferred (f, Z) pairs lie inside Unity 5k training
distribution, or outside? If outside, the residual landing-height error is
focal/depth ambiguity, not model undertraining.

Reads:
  - All Unity 5k camera.json files under TennisDataset_game_5000/
  - PnP results from LSTM_Model/inference_output/pnp_ambiguity_diag.json
    (produced by diagnose_pnp_ambiguity.py; post-mirror-fix values = |Y|)

Prints distribution stats and whether each TrackNet PnP value falls
within Unity's percentile band. No code changes to any other script.

Usage:
    python LSTM_Model/diagnose_focal_distribution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
UNITY_ROOT = REPO / "TennisDataset_game_5000"
PNP_JSON = HERE / "inference_output" / "pnp_ambiguity_diag.json"


def load_unity_cameras(root: Path):
    files = sorted(root.rglob("camera.json"))
    if not files:
        print(f"ERROR: no camera.json under {root}", file=sys.stderr)
        sys.exit(1)
    fx, fy, posY, posZ, fov = [], [], [], [], []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        fx.append(d["fx"])
        fy.append(d["fy"])
        posY.append(d["position"][1])
        posZ.append(d["position"][2])
        fov.append(d.get("fovYDeg", np.nan))
    return {
        "n": len(fx),
        "fx": np.array(fx),
        "fy": np.array(fy),
        "posY": np.array(posY),
        "posZ": np.array(posZ),
        "fov": np.array(fov),
    }


def load_pnp_results(path: Path):
    """Return per-game dict of post-mirror-fix (fx, |Y|, |Z|) from the ITERATIVE column."""
    if not path.exists():
        print(f"ERROR: {path} not found. Run diagnose_pnp_ambiguity.py first.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(path.read_text())
    rows = []
    for r in data:
        if "error" in r or r.get("iterative") is None:
            continue
        it = r["iterative"]
        cam = it["cam_pos"]
        rows.append({
            "label": r["label"],
            "fx": it["focal"],
            # After mirror fix: Y_gl = -Y_iter, so abs()
            "posY_fixed": abs(cam[1]),
            "posZ": cam[2],
        })
    return rows


def pct_rank(value, arr):
    """Percentile position of value inside arr (0..100)."""
    return float((arr < value).mean() * 100.0)


def stats(arr):
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def fmt_stats(name, s):
    return (f"{name:<10}  min={s['min']:8.2f}  p05={s['p05']:8.2f}  p25={s['p25']:8.2f}  "
            f"p50={s['p50']:8.2f}  p75={s['p75']:8.2f}  p95={s['p95']:8.2f}  max={s['max']:8.2f}")


def main():
    print(f"Loading Unity cameras from {UNITY_ROOT} ...", flush=True)
    U = load_unity_cameras(UNITY_ROOT)
    print(f"  loaded {U['n']} camera.json files\n")

    # Unity arrays
    U_fx = U["fx"]
    U_absY = np.abs(U["posY"])           # should already be positive, but safe
    U_absZ = np.abs(U["posZ"])
    U_ZoverF = U_absZ / U_fx

    print("=" * 100)
    print("UNITY 5k TRAINING DISTRIBUTION (n=%d)" % U["n"])
    print("=" * 100)
    print(fmt_stats("fx",    stats(U_fx)))
    print(fmt_stats("|posY|", stats(U_absY)))
    print(fmt_stats("|posZ|", stats(U_absZ)))
    print(fmt_stats("Z/f",   stats(U_ZoverF)))

    # TrackNet PnP
    print("\n" + "=" * 100)
    print("TRACKNET PnP VALUES (per-game, post-mirror-fix)")
    print("=" * 100)
    rows = load_pnp_results(PNP_JSON)
    print(f"{'game':<10}{'fx':>8}{'|posY|':>10}{'|posZ|':>10}{'Z/f':>10}    "
          f"{'fx pct':>8}{'Y pct':>8}{'Z pct':>8}{'Z/f pct':>10}    {'verdict'}")
    print("-" * 100)
    outs = []
    for r in rows:
        fx = r["fx"]
        ay = r["posY_fixed"]
        az = abs(r["posZ"])
        zf = az / fx
        p_fx = pct_rank(fx, U_fx)
        p_y  = pct_rank(ay, U_absY)
        p_z  = pct_rank(az, U_absZ)
        p_zf = pct_rank(zf, U_ZoverF)
        flags = []
        for name, p in [("fx", p_fx), ("|Y|", p_y), ("|Z|", p_z), ("Z/f", p_zf)]:
            if p < 5 or p > 95:
                flags.append(name)
        verdict = "OUT-OF-DIST: " + ",".join(flags) if flags else "inside"
        outs.append(dict(r=r, fx=fx, ay=ay, az=az, zf=zf,
                         p_fx=p_fx, p_y=p_y, p_z=p_z, p_zf=p_zf, verdict=verdict))
        print(f"{r['label']:<10}{fx:>8.0f}{ay:>10.2f}{az:>10.2f}{zf:>10.4f}    "
              f"{p_fx:>7.1f}%{p_y:>7.1f}%{p_z:>7.1f}%{p_zf:>9.1f}%    {verdict}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    n_out = sum(1 for o in outs if o["verdict"] != "inside")
    print(f"  Games with at least one parameter outside Unity's [5%, 95%] band: {n_out} / {len(outs)}")

    # Z/f specifically — the hypothesis from the last message
    unity_zf_p05, unity_zf_p95 = np.percentile(U_ZoverF, [5, 95])
    pnp_zf = np.array([o["zf"] for o in outs])
    n_zf_out = int(((pnp_zf < unity_zf_p05) | (pnp_zf > unity_zf_p95)).sum())
    print(f"  Z/f band (Unity 5%-95%): [{unity_zf_p05:.4f}, {unity_zf_p95:.4f}]")
    print(f"  PnP Z/f range           : [{pnp_zf.min():.4f}, {pnp_zf.max():.4f}]")
    print(f"  PnP Z/f outside Unity band: {n_zf_out} / {len(pnp_zf)}")

    print()
    if n_zf_out >= len(pnp_zf) * 0.5:
        print("  -> PnP Z/f ratios lie OUTSIDE Unity's training band.")
        print("     The focal/depth ambiguity IS a real driver of residual depth error.")
        print("     Fix: constrain the focal sweep to Unity's range, or seed solvePnP with")
        print("     a focal drawn from Unity's distribution.")
    elif n_zf_out == 0:
        print("  -> PnP Z/f ratios lie inside Unity's training band.")
        print("     Focal/depth mismatch is NOT the driver. Look elsewhere for landing-height gap.")
    else:
        print("  -> PnP Z/f partially overlaps Unity's band.")
        print("     Focal prior may still help on the outliers; check per-game reproj vs. Z/f.")

    # Per-game correlation hint: would be best read against eval metrics, but
    # we can at least show which games are most distant from Unity center.
    print("\n  Games ranked by distance from Unity Z/f median (largest first):")
    med_zf = np.median(U_ZoverF)
    ranked = sorted(outs, key=lambda o: abs(o["zf"] - med_zf), reverse=True)
    for o in ranked:
        print(f"    {o['r']['label']:<8}  Z/f={o['zf']:.4f}  (Unity median {med_zf:.4f}, "
              f"delta={o['zf']-med_zf:+.4f})")


if __name__ == "__main__":
    main()
