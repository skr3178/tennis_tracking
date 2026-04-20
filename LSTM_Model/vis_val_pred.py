"""Visualize predicted vs GT trajectories on validation sequences."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from data.dataset import BallTrajectoryDataset, pad_collate
from pipeline import WhereIsTheBall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints_5k_v2/best.pt")
    p.add_argument("--data_root", default="/media/skr/storage/ten_bad/paper_npz_rev1")
    p.add_argument("--split_subdir", default="ours_game_5000_trimmed")
    p.add_argument("--split", default="val")
    p.add_argument("--n_seqs", type=int, default=5)
    p.add_argument("--out_dir", default="inference_output/val_vis")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    ds = BallTrajectoryDataset(
        root=args.data_root, split_subdir=args.split_subdir, split=args.split,
        uv_noise_sigma_px=0.0,
    )

    net = WhereIsTheBall().to(device)
    state = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(state["model_state"])
    net.eval()
    print(f"Loaded {args.ckpt} (epoch {state.get('epoch','?')}), |{args.split}|={len(ds)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Pick sequences spread across the val set
    indices = np.linspace(0, len(ds) - 1, args.n_seqs, dtype=int)

    viewer_data = {}
    for count, idx in enumerate(indices):
        sample = ds[idx]
        P = sample["P"].unsqueeze(0).to(device)
        lengths = sample["length"].unsqueeze(0).to(device)
        xyz_gt = sample["xyz"].numpy()

        with torch.no_grad():
            out = net(P, lengths=lengths)
        xyz_pred = out["xyz_final"][0, :int(sample["length"])].cpu().numpy()

        viewer_data[str(count)] = {
            "pred_refined": xyz_pred.tolist(),
            "ground_truth": xyz_gt.tolist(),
            "name": sample["name"],
        }

        err = np.linalg.norm(xyz_pred - xyz_gt, axis=1)
        print(f"  seq {count} ({sample['name']}): {len(xyz_gt)} frames, "
              f"mean_err={err.mean():.3f}m, max_err={err.max():.3f}m")

    # Save viewer JS
    viewer_path = os.path.join(args.out_dir, "val_pred_vs_gt.js")
    with open(viewer_path, "w") as f:
        f.write("var data = ")
        json.dump(viewer_data, f)
        f.write(";\n")
    print(f"Saved {viewer_path}")

    # Also save a matplotlib plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(args.n_seqs, 3, figsize=(15, 4 * args.n_seqs))
    if args.n_seqs == 1:
        axes = axes[np.newaxis, :]

    for i, key in enumerate(viewer_data):
        d = viewer_data[key]
        gt = np.array(d["ground_truth"])
        pred = np.array(d["pred_refined"])
        name = d["name"]

        # Top-down (X-Z)
        ax = axes[i, 0]
        ax.plot(gt[:, 0], gt[:, 2], 'b-o', markersize=2, label='GT', alpha=0.7)
        ax.plot(pred[:, 0], pred[:, 2], 'r-x', markersize=2, label='Pred', alpha=0.7)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{name} — Top-down")
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Side view (Z-Y)
        ax = axes[i, 1]
        ax.plot(gt[:, 2], gt[:, 1], 'b-o', markersize=2, label='GT', alpha=0.7)
        ax.plot(pred[:, 2], pred[:, 1], 'r-x', markersize=2, label='Pred', alpha=0.7)
        ax.set_xlabel("Z (m)")
        ax.set_ylabel("Y height (m)")
        ax.set_title(f"{name} — Side view")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Height over time
        ax = axes[i, 2]
        frames = np.arange(len(gt))
        ax.plot(frames, gt[:, 1], 'b-', label='GT height', alpha=0.7)
        ax.plot(frames, pred[:, 1], 'r--', label='Pred height', alpha=0.7)
        err = np.linalg.norm(pred - gt, axis=1)
        ax.fill_between(frames, 0, err, alpha=0.15, color='gray', label='3D error')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Height (m)")
        ax.set_title(f"{name} — Height + error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Validation: Predicted vs GT (epoch {state.get('epoch','?')}, NRMSE_dist={state.get('val_metrics',{}).get('nrmse_distance',0):.4f})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "val_pred_vs_gt.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
