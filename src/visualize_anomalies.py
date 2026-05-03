"""
Visualisation Dashboard - Astro-Anomaly-Detector
=================================================
Generates a comprehensive visual analysis of the anomaly detection
results, including:

  1. Anomaly score distribution (histogram + KDE)
  2. Score scatter: MSE vs SSIM-delta coloured by anomaly label
  3. Top-K anomaly tile gallery (multi-panel mosaic)
  4. Normal vs Anomaly comparison grid
  5. Per-channel reconstruction error heatmaps for top anomaly

Run AFTER detect_anomalies.py has produced the anomaly_scores.csv.

Usage:
    python src/visualize_anomalies.py
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch
from torch.utils.data import DataLoader

# -- Project imports --------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.autoencoder import AstroAutoencoder
from src.data.dataset        import AstroTileDataset, percentile_normalise
from src.utils.helpers       import load_config, set_seed, get_device, load_checkpoint, get_data_dirs

# --- Colour Palette ------------------------------------------
BG      = "#0D1117"
SURFACE = "#161B22"
ACCENT  = "#F97316"
BLUE    = "#3B82F6"
GREEN   = "#22C55E"
RED     = "#EF4444"
TEXT    = "#E6EDF3"


def _apply_dark_theme(ax, title=""):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.spines[:].set_color("#30363D")
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, pad=6)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


# -------------------------------------------------------------
#  1. Score Distribution (histogram + percentile line)
# -------------------------------------------------------------

def plot_score_distribution(df: pd.DataFrame, threshold: float, out_path: str):
    scores = df["Anomaly_Score"].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(BG)

    # - Left: histogram -
    ax = axes[0]
    _apply_dark_theme(ax, "Anomaly Score Distribution")
    ax.hist(scores, bins=80, color=BLUE, alpha=0.8, edgecolor="none")
    ax.axvline(threshold, color=ACCENT, lw=2, ls="--", label=f"Threshold = {threshold:.4f}")
    n_anom = int((scores > threshold).sum())
    ax.text(threshold * 1.03, ax.get_ylim()[1] * 0.8,
            f"{n_anom} anomalies\n({100*n_anom/len(scores):.1f}%)",
            color=ACCENT, fontsize=10)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.legend(facecolor=SURFACE, labelcolor=TEXT)

    # - Right: MSE vs SSIM scatter -
    ax = axes[1]
    _apply_dark_theme(ax, "MSE vs SSIM-Delta (coloured by anomaly score)")
    sc = ax.scatter(
        df["MSE"], df["SSIM_Delta"],
        c=df["Anomaly_Score"], cmap="plasma", s=4, alpha=0.6
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    cbar.set_label("Anomaly Score", color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    ax.set_xlabel("MSE")
    ax.set_ylabel("SSIM-Delta")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [1] Score analysis plot -> {out_path}")


# -------------------------------------------------------------
#  2. Top-K Anomaly Mosaic Gallery
# -------------------------------------------------------------

def plot_anomaly_gallery(
    top_tiles: list,      # list of dicts: {fname, original(C,H,W), recon, score, mse, ssim}
    out_path: str,
    n_cols: int = 5,
):
    n = len(top_tiles)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    fig.patch.set_facecolor(BG)
    axes = np.array(axes).reshape(-1)

    for i, info in enumerate(top_tiles):
        ax  = axes[i]
        rgb = np.stack([info["original"][c] for c in [0, 1, min(2, info["original"].shape[0]-1)]], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        ax.imshow(rgb, origin="upper")
        ax.set_title(
            f"Score: {info['score']:.3f}\n{info['fname'][:18]}",
            color=ACCENT, fontsize=8, pad=2
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Anomalous Tiles - RGB Preview (Ch 0,1,2)", color=TEXT, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [2] Anomaly gallery   -> {out_path}")


# -------------------------------------------------------------
#  3. Normal vs Anomaly Comparison Grid
# -------------------------------------------------------------

def plot_normal_vs_anomaly(normal_tiles: list, anomaly_tiles: list, out_path: str, n_each: int = 6):
    n_each  = min(n_each, len(normal_tiles), len(anomaly_tiles))
    fig, axes = plt.subplots(2, n_each, figsize=(n_each * 3, 7))
    fig.patch.set_facecolor(BG)

    labels = ["NORMAL", "ANOMALY"]
    tile_groups = [normal_tiles[:n_each], anomaly_tiles[:n_each]]
    title_colors = [GREEN, RED]

    for row, (group, label, tcolor) in enumerate(zip(tile_groups, labels, title_colors)):
        for col, info in enumerate(group):
            ax  = axes[row][col]
            rgb = np.stack([info["original"][c] for c in [0, 1, min(2, info["original"].shape[0]-1)]], axis=-1)
            rgb = np.clip(rgb, 0, 1)
            ax.imshow(rgb, origin="upper")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, color=tcolor, fontsize=12, labelpad=4)
            ax.set_title(f"{info['score']:.3f}", color=tcolor, fontsize=9)

    fig.suptitle("Normal Tiles vs Anomalous Tiles", color=TEXT, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [3] Normal vs anomaly -> {out_path}")


# -------------------------------------------------------------
#  4. Per-Channel Reconstruction Error for #1 Anomaly
# -------------------------------------------------------------

def plot_per_channel_error(tile_info: dict, out_path: str):
    n_ch    = tile_info["original"].shape[0]
    ch_labels = [f"Channel {i}" for i in range(n_ch)]

    fig, axes = plt.subplots(3, n_ch, figsize=(n_ch * 4, 11))
    fig.patch.set_facecolor(BG)

    row_labels = ["Original", "Reconstructed", "Abs Error"]

    for ch in range(n_ch):
        orig_ch  = tile_info["original"][ch]
        recon_ch = tile_info["recon"][ch]
        err_ch   = np.abs(orig_ch - recon_ch)

        for row, (img, rlab, cmap) in enumerate(zip(
            [orig_ch, recon_ch, err_ch],
            row_labels,
            ["viridis", "viridis", "hot"]
        )):
            ax = axes[row][ch]
            im = ax.imshow(img, cmap=cmap, origin="upper", vmin=0)
            plt.colorbar(im, ax=ax, fraction=0.046)
            _apply_dark_theme(ax)
            ax.set_title(f"{rlab} - {ch_labels[ch]}", color=TEXT, fontsize=9)
            ax.axis("off")

    fig.suptitle(
        f"Per-Channel Analysis: TOP Anomaly - {tile_info['fname']}\n"
        f"Score: {tile_info['score']:.4f}  |  MSE: {tile_info['mse']:.4f}",
        color=ACCENT, fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [4] Per-channel error -> {out_path}")


# -------------------------------------------------------------
#  5. Anomaly Score Time-Series  (the spike graph)
# -------------------------------------------------------------

def plot_anomaly_timeseries(df: pd.DataFrame, threshold: float, out_path: str):
    """
    Recreates the classic anomaly detection spike graph:
      - Smooth blue signal line (rolling mean of normal scores = flat baseline)
      - Individual vertical red spikes at anomaly positions
      - Red dot on the highest anomaly (like the professor's reference image)
      - Orange dashed threshold line
    """
    import matplotlib.patches as mpatches

    # Sort by tile index so x-axis is sequential
    df_sorted = df.sort_values("Filename").reset_index(drop=True)

    # Use raw MSE — gives the FLATTEST normal baseline and tallest anomaly spikes
    # Normal MSE ~0.004 vs anomaly MSE ~0.03-0.05  = clearly visible spikes
    scores   = df_sorted["MSE"].values
    is_anom  = df_sorted["Is_Anomaly"].values
    mse_thr  = float(np.percentile(scores, 97))   # recompute threshold on MSE
    tile_idx = np.arange(len(scores))

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Smooth baseline: rolling mean of ALL scores (looks like
    #    the wavy-but-bounded signal in the professor's graph) ──
    window = 15
    baseline = np.convolve(scores, np.ones(window)/window, mode='same')

    # Draw the smooth blue signal line
    ax.plot(tile_idx, baseline,
            color="#1E90FF", linewidth=1.4, alpha=0.9, zorder=2,
            label="Reconstruction Error")

    # ── Anomaly spikes: draw tall vertical lines at each anomaly ──
    anom_positions = tile_idx[is_anom]
    anom_scores    = scores[is_anom]

    for xi, yi in zip(anom_positions, anom_scores):
        ax.vlines(xi, ymin=mse_thr * 0.5, ymax=yi,
                  colors="#E63946", linewidth=1.5, alpha=0.85, zorder=4)

    # Red dots on top of each spike
    ax.scatter(anom_positions, anom_scores,
               color="#E63946", s=18, zorder=5, linewidths=0,
               label=f"Anomaly ({is_anom.sum()} detected)")

    # ── Threshold line ──────────────────────────────────────────
    ax.axhline(mse_thr, color="#FF8C00", linewidth=1.8,
               linestyle="--", zorder=3, alpha=0.9,
               label=f"Threshold (P97 MSE) = {mse_thr:.4f}")

    # ── Annotate the single biggest spike (like professor's graph) ─
    top_i     = np.argmax(scores)
    top_score = scores[top_i]
    offset_x  = min(top_i + len(scores)//12, len(scores)-1)
    ax.annotate(
        "Anomaly",
        xy=(top_i, top_score),
        xytext=(offset_x, top_score * 1.05),
        arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.8),
        color="#E63946", fontsize=12, fontweight="bold",
        ha="center"
    )
    # Red dot on that specific point (larger, more visible)
    ax.scatter([top_i], [top_score],
               color="#E63946", s=80, zorder=6, linewidths=0)

    # ── Axis styling (clean white background like professor's) ──
    ax.set_xlabel("Tile Index", fontsize=13, color="black")
    ax.set_ylabel("MSE (Reconstruction Error)", fontsize=12, color="black")
    ax.set_title("Anomaly Detection - Reconstruction Error per Tile",
                 fontsize=14, color="black", pad=12, fontweight="bold")
    ax.tick_params(colors="black", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
    ax.set_xlim(-20, len(scores) + 20)
    ax.set_ylim(0, top_score * 1.25)

    # Grid (light, like professor's background)
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", fontsize=10,
              facecolor="white", edgecolor="#CCCCCC", labelcolor="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  [5] Anomaly time-series -> {out_path}")


# -------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------

@torch.no_grad()
def get_tile_info(model, tile_path: str, device: torch.device) -> dict:
    """Load one tile, run autoencoder, return dict of arrays."""
    import numpy as np
    data = np.load(tile_path).astype(np.float32)
    norm = percentile_normalise(data)
    t    = torch.from_numpy(norm).unsqueeze(0).to(device)
    recon, _ = model(t)
    orig_np  = norm
    recon_np = recon.squeeze(0).cpu().numpy()
    mse      = float(np.mean((orig_np - recon_np) ** 2))
    return {
        "fname":    os.path.basename(tile_path),
        "original": orig_np,
        "recon":    recon_np,
        "mse":      mse,
        "score":    mse,   # placeholder; will be overridden from CSV
        "ssim":     0.0,
    }


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    cfg      = load_config(cfg_path)

    set_seed(42)
    device = get_device()

    o_cfg    = cfg["output"]
    a_cfg    = cfg["anomaly"]
    m_cfg    = cfg["model"]
    d_cfg    = cfg["data"]

    report_dir = os.path.join(PROJECT_ROOT, o_cfg["report_dir"])
    map_dir    = os.path.join(PROJECT_ROOT, o_cfg["anomaly_map_dir"])
    os.makedirs(report_dir, exist_ok=True)

    # -- Load score CSV ----------------------------------------
    csv_path = os.path.join(PROJECT_ROOT, o_cfg["anomaly_csv"])
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found. Run detect_anomalies.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    threshold = float(np.percentile(df["Anomaly_Score"].values, a_cfg["threshold_percentile"]))

    # -- Load model --------------------------------------------
    model = AstroAutoencoder(
        n_channels    = d_cfg["num_channels"],
        base_channels = m_cfg["base_channels"],
        latent_dim    = m_cfg["latent_dim"],
    ).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, o_cfg["checkpoint_dir"], o_cfg["best_model_name"])
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    # -- Val dir ----------------------------------------------
    _, val_dir = get_data_dirs(cfg)

    # -- Build tile info list for top-N anomalies & N normals -
    top_n   = min(20, int(df["Is_Anomaly"].sum()))
    top_df  = df[df["Is_Anomaly"]].head(top_n)
    bot_df  = df[~df["Is_Anomaly"]].tail(top_n)

    def build_info_list(sub_df):
        infos = []
        for _, row in sub_df.iterrows():
            fpath = os.path.join(val_dir, row["Filename"])
            if not os.path.exists(fpath):
                continue
            info = get_tile_info(model, fpath, device)
            info["score"] = row["Anomaly_Score"]
            info["mse"]   = row["MSE"]
            info["ssim"]  = row["SSIM_Delta"]
            infos.append(info)
        return infos

    print(f"\n{'='*60}")
    print("  Astro-Anomaly-Detector - Visualisation Dashboard")
    print(f"{'='*60}")
    print(f"  Building data for top-{top_n} anomalies & normals ...\n")

    anomaly_tiles = build_info_list(top_df)
    normal_tiles  = build_info_list(bot_df)

    # -- 1. Score distribution ---------------------------------
    plot_score_distribution(df, threshold,
        os.path.join(report_dir, "1_score_analysis.png"))

    # -- 2. Anomaly gallery ------------------------------------
    if anomaly_tiles:
        plot_anomaly_gallery(anomaly_tiles,
            os.path.join(report_dir, "2_anomaly_gallery.png"), n_cols=5)

    # -- 3. Normal vs Anomaly ----------------------------------
    if anomaly_tiles and normal_tiles:
        plot_normal_vs_anomaly(normal_tiles, anomaly_tiles,
            os.path.join(report_dir, "3_normal_vs_anomaly.png"), n_each=6)

    # -- 4. Per-channel error for #1 anomaly -------------------
    if anomaly_tiles:
        plot_per_channel_error(anomaly_tiles[0],
            os.path.join(report_dir, "4_channel_error_top1.png"))

    # -- 5. Anomaly time-series spike graph --------------------
    plot_anomaly_timeseries(df, threshold,
        os.path.join(report_dir, "5_anomaly_timeseries.png"))

    print(f"\n{'='*60}")
    print("  All visualisations saved to:")
    print(f"  {report_dir}")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    main()
