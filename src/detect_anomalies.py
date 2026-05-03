"""
Anomaly Detection Script - Astro-Anomaly-Detector
===================================================
Loads the trained autoencoder and scores EVERY tile in the
validation set by its reconstruction error.

Outputs:
  |-- outputs/reports/anomaly_scores.csv       - full per-tile score table
  |-- outputs/reports/anomaly_score_distribution.png  - histogram + threshold
  |-- outputs/reports/anomaly_summary.txt      - human-readable summary
  +-- outputs/anomaly_maps/<tile_name>.png     - per-tile visual  (top-N only)

Anomaly Score = alpha x MSE  +  (1-alpha) x (1 - SSIM-proxy)
  Higher score -> more anomalous

Usage:
    python src/detect_anomalies.py
"""

import os
import sys
import csv

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# -- Project imports --------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.autoencoder import AstroAutoencoder, reconstruction_error_map
from src.data.dataset        import AstroTileDataset
from src.utils.helpers       import (
    load_config, set_seed, get_device,
    load_checkpoint, get_data_dirs
)


# -------------------------------------------------------------
#  SSIM-proxy: local variance similarity  (fast, no extra dep)
# -------------------------------------------------------------

def _local_var(x: torch.Tensor, kernel: int = 8) -> torch.Tensor:
    """
    Approximate local variance via avg-pooling.
    x shape: (B, C, H, W) -> returns (B,) scalar per sample.
    """
    mu     = torch.nn.functional.avg_pool2d(x,  kernel, stride=kernel)
    mu_sq  = torch.nn.functional.avg_pool2d(x*x, kernel, stride=kernel)
    var    = (mu_sq - mu**2).clamp(min=0)
    return var.mean(dim=(1, 2, 3))


def ssim_proxy_score(original: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """
    Structural similarity proxy:
      1 - (2 * cov(orig,recon) + C) / (var_orig + var_recon + C)
    Higher -> less similar -> more anomalous.
    C stabilises the division.
    """
    C   = 1e-5
    var_o = _local_var(original)
    var_r = _local_var(recon)
    cov   = _local_var((original + recon) / 2) - (var_o + var_r) / 4
    ssim  = (2 * cov + C) / (var_o + var_r + C)
    return 1.0 - ssim.clamp(0, 1)   # anomaly-direction: higher = worse


# -------------------------------------------------------------
#  Per-tile scoring
# -------------------------------------------------------------

@torch.no_grad()
def score_tiles(model, loader, device, ssim_weight: float = 0.5):
    """
    Pass all tiles through the autoencoder and compute anomaly scores.
    Returns:
        filenames  : list[str]
        scores     : np.ndarray   (N,)
        mse_scores : np.ndarray   (N,)
        ssim_scores: np.ndarray   (N,)
        error_maps : list[ np.ndarray (H,W) ]  - spatial heat-maps
        originals  : list[ np.ndarray (C,H,W) ]
        recons     : list[ np.ndarray (C,H,W) ]
    """
    model.eval()
    all_fnames  = []
    all_scores  = []
    all_mse     = []
    all_ssim    = []
    all_emaps   = []
    all_originals = []
    all_recons    = []

    for tiles, fnames in tqdm(loader, desc="Scoring tiles"):
        tiles = tiles.to(device, non_blocking=True)

        recon, _ = model(tiles)

        # Per-sample MSE
        mse = (tiles - recon).pow(2).mean(dim=(1, 2, 3))          # (B,)
        # Per-sample SSIM-proxy
        ssim = ssim_proxy_score(tiles, recon)                       # (B,)
        # Combined score
        combined = (1 - ssim_weight) * mse + ssim_weight * ssim    # (B,)

        # Spatial error map (first channel as representative)
        emap = reconstruction_error_map(tiles, recon)               # (B, H, W)

        all_fnames.extend(list(fnames))
        all_scores.extend(combined.cpu().numpy().tolist())
        all_mse.extend(   mse.cpu().numpy().tolist())
        all_ssim.extend(  ssim.cpu().numpy().tolist())

        # Save originals and recons for top-N vizualisation (CPU)
        for i in range(tiles.size(0)):
            all_emaps.append(    emap[i].cpu().numpy())
            all_originals.append(tiles[i].cpu().numpy())
            all_recons.append(   recon[i].cpu().numpy())

    return (
        all_fnames,
        np.array(all_scores, dtype=np.float32),
        np.array(all_mse,    dtype=np.float32),
        np.array(all_ssim,   dtype=np.float32),
        all_emaps,
        all_originals,
        all_recons,
    )


# -------------------------------------------------------------
#  Visualisation Helpers
# -------------------------------------------------------------

def _ch_to_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Convert (C, H, W) -> (H, W, 3) for display.
    Uses channels 0,1,2 as R,G,B (drops ch3 if present).
    """
    rgb = np.stack([arr[min(c, arr.shape[0]-1)] for c in [0, 1, 2]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return rgb


def save_anomaly_tile(fname, original, recon, emap, score, mse, ssim_s, out_dir):
    """Save a 4-panel figure: Original | Reconstructed | Error Map | Channel-0 diff."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.patch.set_facecolor("#0D1117")

    titles = ["Original (Ch0-2)", "Reconstructed", "Error Map", "Diff (Ch0)"]
    imgs   = [
        _ch_to_rgb(original),
        _ch_to_rgb(recon),
        emap,
        np.abs(original[0] - recon[0]),
    ]
    cmaps  = ["viridis", "viridis", "hot", "inferno"]

    for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(img, cmap=cmap, origin="upper", vmin=0)
        ax.set_title(title, color="white", fontsize=11, pad=4)
        ax.axis("off")

    sup = (f"Tile: {fname}  |  Anomaly Score: {score:.4f}  "
           f"|  MSE: {mse:.4f}  |  SSIM-Delta: {ssim_s:.4f}")
    fig.suptitle(sup, color="#F97316", fontsize=10, y=1.01)

    plt.tight_layout()
    out_path = os.path.join(out_dir, fname.replace(".npy", "_anomaly.png"))
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def save_score_distribution(scores, threshold, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    ax.hist(scores, bins=80, color="#3B82F6", alpha=0.8, edgecolor="none", label="All tiles")
    ax.axvline(threshold, color="#F97316", linewidth=2,
               linestyle="--", label=f"Threshold = {threshold:.4f}")

    n_anomaly = int((scores > threshold).sum())
    ax.text(threshold * 1.02, ax.get_ylim()[1] * 0.85,
            f"{n_anomaly} anomalies\n({100*n_anomaly/len(scores):.1f}%)",
            color="#F97316", fontsize=11)

    ax.set_xlabel("Anomaly Score", color="white")
    ax.set_ylabel("Count",         color="white")
    ax.set_title("Anomaly Score Distribution", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363D")
    ax.legend(facecolor="#161B22", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Distribution plot -> {out_path}")


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    cfg      = load_config(cfg_path)

    set_seed(42)
    device = get_device()

    # -- Load val dataset -------------------------------------
    _, val_dir = get_data_dirs(cfg)
    val_ds  = AstroTileDataset(val_dir)
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
    )

    # -- Load model -------------------------------------------
    m_cfg    = cfg["model"]
    d_cfg    = cfg["data"]
    o_cfg    = cfg["output"]
    a_cfg    = cfg["anomaly"]

    model = AstroAutoencoder(
        n_channels    = d_cfg["num_channels"],
        base_channels = m_cfg["base_channels"],
        latent_dim    = m_cfg["latent_dim"],
    ).to(device)

    ckpt_path = os.path.join(PROJECT_ROOT, o_cfg["checkpoint_dir"], o_cfg["best_model_name"])
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("        Please run train_autoencoder.py first.")
        sys.exit(1)

    load_checkpoint(model, ckpt_path, device)
    model.eval()

    # -- Score ------------------------------------------------
    print(f"\n{'='*60}")
    print("  Astro-Anomaly-Detector - Anomaly Scoring")
    print(f"{'='*60}")
    print(f"  Scoring {len(val_ds)} tiles ...\n")

    fnames, scores, mse_s, ssim_s, emaps, originals, recons = score_tiles(
        model, val_loader, device,
        ssim_weight = a_cfg["ssim_weight"],
    )

    # -- Threshold --------------------------------------------
    threshold   = float(np.percentile(scores, a_cfg["threshold_percentile"]))
    is_anomaly  = scores > threshold
    n_anomalies = int(is_anomaly.sum())

    print(f"\n  Total tiles scored : {len(scores)}")
    print(f"  Threshold (@{a_cfg['threshold_percentile']}th pct) : {threshold:.5f}")
    print(f"  Anomalies detected : {n_anomalies}  ({100*n_anomalies/len(scores):.1f}%)")

    # -- Save CSV ---------------------------------------------
    report_dir  = os.path.join(PROJECT_ROOT, o_cfg["report_dir"])
    map_dir     = os.path.join(PROJECT_ROOT, o_cfg["anomaly_map_dir"])
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(map_dir,    exist_ok=True)

    csv_path = os.path.join(PROJECT_ROOT, o_cfg["anomaly_csv"])
    df = pd.DataFrame({
        "Filename":      fnames,
        "Anomaly_Score": scores,
        "MSE":           mse_s,
        "SSIM_Delta":    ssim_s,
        "Is_Anomaly":    is_anomaly,
    }).sort_values("Anomaly_Score", ascending=False)

    df.to_csv(csv_path, index=False)
    print(f"\n  Full score table -> {csv_path}")

    # -- Distribution plot ------------------------------------
    dist_plot = os.path.join(PROJECT_ROOT, o_cfg["summary_plot"])
    save_score_distribution(scores, threshold, dist_plot)

    # -- Save top-N anomaly visuals ---------------------------
    top_n  = min(a_cfg["top_n_anomalies"], n_anomalies)
    top_idx = np.argsort(scores)[::-1][:top_n]

    print(f"\n  Saving top-{top_n} anomaly tile visualisations ...")
    saved_paths = []
    for rank, idx in enumerate(tqdm(top_idx, desc="Saving visuals"), start=1):
        path = save_anomaly_tile(
            fname    = fnames[idx],
            original = originals[idx],
            recon    = recons[idx],
            emap     = emaps[idx],
            score    = float(scores[idx]),
            mse      = float(mse_s[idx]),
            ssim_s   = float(ssim_s[idx]),
            out_dir  = map_dir,
        )
        saved_paths.append(path)

    # -- Text summary -----------------------------------------
    summary_path = os.path.join(report_dir, "anomaly_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  ASTRO-ANOMALY-DETECTOR - DETECTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Total tiles scored   : {len(scores)}\n")
        f.write(f"  Threshold (@{a_cfg['threshold_percentile']}th pct): {threshold:.6f}\n")
        f.write(f"  Score metric         : {a_cfg['score_metric']}\n")
        f.write(f"  Anomalies detected   : {n_anomalies} ({100*n_anomalies/len(scores):.2f}%)\n\n")
        f.write("  --- TOP-20 MOST ANOMALOUS TILES ---\n\n")
        f.write(f"  {'Rank':<5} {'Filename':<30} {'Score':>10} {'MSE':>10} {'SSIM-Delta':>10}\n")
        f.write("  " + "-" * 68 + "\n")
        for rank, idx in enumerate(top_idx[:20], start=1):
            f.write(
                f"  {rank:<5} {fnames[idx]:<30} "
                f"{scores[idx]:>10.5f} {mse_s[idx]:>10.5f} {ssim_s[idx]:>10.5f}\n"
            )

    print(f"  Summary report    -> {summary_path}")
    print(f"\n{'='*60}")
    print("  Anomaly detection complete!")
    print(f"  Next step -> run: python src/visualize_anomalies.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
