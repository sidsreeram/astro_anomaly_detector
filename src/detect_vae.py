"""
Anomaly Detection Script - Astro-Anomaly-Detector (VAE Version)
===============================================================
Loads the trained VAE and scores EVERY tile in validation set.

Outputs:
  |-- outputs/reports/anomaly_scores_vae.csv       - full per-tile score table
  |-- outputs/reports/anomaly_score_distribution_vae.png

Score = (0.5 * MSE) + (0.5 * SSIM-proxy) + (KLD_weight * KLD)
  Higher score -> more anomalous

Usage:
    python src/detect_vae.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vae import ResNetVAE, reconstruction_error_map
from src.data.dataset import AstroTileDataset
from src.utils.helpers import load_config, set_seed, get_device, load_checkpoint, get_data_dirs
from src.detect_anomalies import ssim_proxy_score, save_anomaly_tile, save_score_distribution

@torch.no_grad()
def score_tiles_vae(model, loader, device, ssim_weight=0.5, kld_weight=0.001):
    """Pass all tiles through the VAE and compute anomaly scores using expected value mapping."""
    model.eval()
    all_fnames, all_scores, all_mse, all_ssim, all_kld = [], [], [], [], []
    all_emaps, all_originals, all_recons = [], [], []

    for tiles, fnames in tqdm(loader, desc="Scoring tiles"):
        tiles = tiles.to(device, non_blocking=True)

        # Deterministic Inference for Scoring (skip sampling)
        mu, logvar = model.encode(tiles)
        recon = model.decode(mu) # bypass reparameterize for consistent scoring

        # Metrics
        mse  = (tiles - recon).pow(2).mean(dim=(1, 2, 3))
        ssim = ssim_proxy_score(tiles, recon)
        
        # KLD
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Combined score
        combined = (1 - ssim_weight) * mse + ssim_weight * ssim + (kld_weight * kld)
        emap = reconstruction_error_map(tiles, recon)

        all_fnames.extend(list(fnames))
        all_scores.extend(combined.cpu().numpy().tolist())
        all_mse.extend(mse.cpu().numpy().tolist())
        all_ssim.extend(ssim.cpu().numpy().tolist())
        all_kld.extend(kld.cpu().numpy().tolist())

        for i in range(tiles.size(0)):
            all_emaps.append(emap[i].cpu().numpy())
            all_originals.append(tiles[i].cpu().numpy())
            all_recons.append(recon[i].cpu().numpy())

    return (
        all_fnames,
        np.array(all_scores, dtype=np.float32),
        np.array(all_mse, dtype=np.float32),
        np.array(all_ssim, dtype=np.float32),
        np.array(all_kld, dtype=np.float32),
        all_emaps, all_originals, all_recons
    )

def main():
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    cfg = load_config(cfg_path)

    set_seed(42)
    device = get_device()
    _, val_dir = get_data_dirs(cfg)
    
    val_ds  = AstroTileDataset(val_dir)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    o_cfg = cfg["output"]
    a_cfg = cfg["anomaly"]
    t_cfg = cfg["training"]

    model = ResNetVAE(
        in_channels   = d_cfg["num_channels"],
        base_channels = m_cfg["base_channels"],
        latent_dim    = m_cfg["latent_dim"],
    ).to(device)

    ckpt_path = os.path.join(PROJECT_ROOT, o_cfg["checkpoint_dir"], o_cfg["best_model_name"])
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] VAE checkpoint not found: {ckpt_path}")
        sys.exit(1)

    load_checkpoint(model, ckpt_path, device)
    
    print("\n" + "="*60)
    print("  Astro-Anomaly-Detector - VAE Anomaly Scoring")
    print("="*60)
    
    kld_weight = t_cfg.get("kld_weight", 0.001)

    fnames, scores, mse_s, ssim_s, kld_s, emaps, originals, recons = score_tiles_vae(
        model, val_loader, device, ssim_weight=a_cfg["ssim_weight"], kld_weight=kld_weight
    )

    threshold   = float(np.percentile(scores, a_cfg["threshold_percentile"]))
    is_anomaly  = scores > threshold
    n_anomalies = int(is_anomaly.sum())

    print(f"\nTotal tiles scored : {len(scores)}")
    print(f"Threshold (@{a_cfg['threshold_percentile']}th pct) : {threshold:.5f}")
    print(f"Anomalies detected : {n_anomalies}  ({100*n_anomalies/len(scores):.1f}%)")

    report_dir = os.path.join(PROJECT_ROOT, o_cfg["report_dir"])
    map_dir    = os.path.join(PROJECT_ROOT, o_cfg["anomaly_map_dir"])
    
    csv_path = os.path.join(PROJECT_ROOT, o_cfg["anomaly_csv"])
    df = pd.DataFrame({
        "Filename":      fnames,
        "Anomaly_Score": scores,
        "MSE":           mse_s,
        "SSIM_Delta":    ssim_s,
        "KLD":           kld_s,
        "Is_Anomaly":    is_anomaly,
    }).sort_values("Anomaly_Score", ascending=False)
    df.to_csv(csv_path, index=False)
    
    dist_plot = os.path.join(PROJECT_ROOT, o_cfg["summary_plot"])
    save_score_distribution(scores, threshold, dist_plot)

    print(f"\nSaving top anomaly tile visualisations...")
    top_n  = min(a_cfg["top_n_anomalies"], n_anomalies)
    top_idx = np.argsort(scores)[::-1][:top_n]
    for idx in tqdm(top_idx, desc="Saving visuals"):
        save_anomaly_tile(fnames[idx], originals[idx], recons[idx], emaps[idx], 
                          float(scores[idx]), float(mse_s[idx]), float(ssim_s[idx]), map_dir)

    print(f"\nDetect Done! Report -> {csv_path}")

if __name__ == "__main__":
    main()
