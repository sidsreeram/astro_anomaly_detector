"""
Training Script - Astro-Anomaly-Detector (VAE/ResNet Version)
=============================================================
Trains a Variational Autoencoder on normal astronomical tile data.
The model learns to reconstruct NORMAL tiles well and maps them
to a standard normal Gaussian prior.

Usage:
    python src/train_vae.py

The script reads all settings from configs/config.yaml.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# -- Project imports --------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vae import ResNetVAE, count_parameters
from src.data.dataset        import get_dataloaders
from src.utils.helpers       import (
    load_config, set_seed, get_device,
    save_checkpoint, load_checkpoint, get_data_dirs
)

# -------------------------------------------------------------
#  Combined Loss: MSE + SSIM-proxy (grad) + KL-Divergence
# -------------------------------------------------------------

class VAELoss(nn.Module):
    def __init__(self, grad_weight: float = 0.1, kld_weight: float = 0.001):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.grad_weight = grad_weight
        self.kld_weight  = kld_weight

    def _gradient_loss(self, original, recon):
        # Sobel-like finite differences
        dx_orig  = original[:, :, :, 1:] - original[:, :, :, :-1]
        dy_orig  = original[:, :, 1:, :] - original[:, :, :-1, :]
        dx_recon = recon[:, :, :, 1:]    - recon[:, :, :, :-1]
        dy_recon = recon[:, :, 1:, :]    - recon[:, :, :-1, :]
        return self.mse(dx_orig, dx_recon) + self.mse(dy_orig, dy_recon)

    def forward(self, original, recon, mu, logvar):
        loss_mse  = self.mse(recon, original)
        loss_grad = self._gradient_loss(original, recon)
        
        # KL Divergence
        loss_kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        total_loss = loss_mse + self.grad_weight * loss_grad + self.kld_weight * loss_kld
        return total_loss, loss_mse.item(), loss_kld.item()

# -------------------------------------------------------------
#  Training Loop
# -------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_mse  = 0.0
    total_kld  = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:>3d} [Train]", leave=False)
    for tiles, _ in pbar:
        tiles = tiles.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        recon, mu, logvar = model(tiles)
        loss, mse_val, kld_val = criterion(tiles, recon, mu, logvar)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse  += mse_val
        total_kld  += kld_val
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.5f}", kld=f"{kld_val:.4f}")

    return total_loss / n_batches, total_mse / n_batches, total_kld / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    max_batches = 10 if epoch == 1 else None
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:>3d} [Val  ]", leave=False)
    for tiles, _ in pbar:
        if max_batches and n_batches >= max_batches:
             break
        tiles = tiles.to(device, non_blocking=True)
        recon, mu, logvar = model(tiles)
        loss, _, _ = criterion(tiles, recon, mu, logvar)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def save_training_curve(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", color="#3B82F6", linewidth=2)
    ax.plot(val_losses,   label="Val Loss",   color="#F97316", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total VAE Loss")
    ax.set_title("ResNet-VAE Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    cfg      = load_config(cfg_path)

    set_seed(42)
    device = get_device()
    train_dir, val_dir = get_data_dirs(cfg)

    print(f"\n{'='*60}")
    print("  Astro-Anomaly-Detector - VAE Training")
    print(f"{'='*60}")

    t_cfg   = cfg["training"]
    d_cfg   = cfg["data"]

    # Limit dataset if data_fraction < 1.0
    import math
    train_count = len(os.listdir(train_dir))
    fraction = t_cfg.get("data_fraction", 1.0)
    max_t = int(train_count * fraction) if fraction < 1.0 else None

    train_loader, val_loader = get_dataloaders(
        train_dir    = train_dir,
        val_dir      = val_dir,
        batch_size   = t_cfg["batch_size"],
        num_workers  = 0,
        max_tiles    = max_t,
    )

    m_cfg = cfg["model"]
    model = ResNetVAE(
        in_channels   = d_cfg["num_channels"],
        base_channels = m_cfg["base_channels"],
        latent_dim    = m_cfg["latent_dim"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr           = t_cfg["learning_rate"],
        weight_decay = t_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max    = t_cfg["epochs"],
        eta_min  = t_cfg["learning_rate"] * 0.01,
    )
    
    kld_weight = t_cfg.get("kld_weight", 0.001)
    criterion = VAELoss(grad_weight=0.1, kld_weight=kld_weight)

    o_cfg       = cfg["output"]
    ckpt_dir    = os.path.join(PROJECT_ROOT, o_cfg["checkpoint_dir"])
    report_dir  = os.path.join(PROJECT_ROOT, o_cfg["report_dir"])
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    best_ckpt  = os.path.join(ckpt_dir, o_cfg["best_model_name"])
    last_ckpt  = os.path.join(ckpt_dir, "last_vae.pth")
    curve_path = os.path.join(report_dir, "training_curve_vae.png")

    start_epoch = 1
    best_val    = float("inf")

    if args.resume and os.path.exists(best_ckpt):
        resumed_epoch, resumed_val = load_checkpoint(model, best_ckpt, device, optimizer)
        start_epoch = resumed_epoch + 1
        best_val    = resumed_val
        for _ in range(resumed_epoch):
            scheduler.step()
        print(f"  >>  Resuming from epoch {resumed_epoch}")

    print(f"  Model parameters : {count_parameters(model):,}")
    print(f"  Latent dim       : {m_cfg['latent_dim']}")
    print(f"  Batch size       : {t_cfg['batch_size']}")
    print(f"  Epochs           : {t_cfg['epochs']}")

    train_hist = [float("nan")] * (start_epoch - 1)
    val_hist   = [float("nan")] * (start_epoch - 1)
    
    # Pre-fill val_hist from existing checkpoints to make the curve continuous
    import glob
    for ckpt_file in glob.glob(os.path.join(ckpt_dir, "vae_epoch*.pth")):
        try:
            data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            e = data.get("epoch")
            v = data.get("val_loss")
            if e and v and e < start_epoch:
                val_hist[e - 1] = v
        except:
            pass
            
    t_start    = time.time()

    for epoch in range(start_epoch, t_cfg["epochs"] + 1):
        t_loss, t_mse, t_kld = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        v_loss = validate(model, val_loader, criterion, device, epoch)

        scheduler.step()
        train_hist.append(t_loss)
        val_hist.append(v_loss)

        elapsed = (time.time() - t_start) / 60
        print(f"Epoch {epoch:>3d}/{t_cfg['epochs']} | Train: {t_loss:.4f} (MSE:{t_mse:.4f} KLD:{t_kld:.4f}) | Val: {v_loss:.4f} | {elapsed:.1f} min")

        if v_loss < best_val:
            best_val = v_loss
            save_checkpoint(model, optimizer, epoch, v_loss, best_ckpt)
            print("  [BEST] New best model saved")

        if epoch % t_cfg["save_every"] == 0:
            periodic = os.path.join(ckpt_dir, f"vae_epoch{epoch:03d}.pth")
            save_checkpoint(model, optimizer, epoch, v_loss, periodic)

    save_checkpoint(model, optimizer, t_cfg["epochs"], v_loss, last_ckpt)
    save_training_curve(train_hist, val_hist, curve_path)

    print(f"\n  Training VAE complete! Best Val Loss: {best_val:.5f}")

if __name__ == "__main__":
    main()
