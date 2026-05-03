"""
Training Script - Astro-Anomaly-Detector
=========================================
Trains a Convolutional Autoencoder on normal astronomical tile data.
The model learns to reconstruct NORMAL tiles well.
Anomalous tiles will have HIGH reconstruction error during inference.

Usage:
    python src/train_autoencoder.py

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

from src.models.autoencoder import AstroAutoencoder, count_parameters
from src.data.dataset        import get_dataloaders
from src.utils.helpers       import (
    load_config, set_seed, get_device,
    save_checkpoint, load_checkpoint, get_data_dirs
)


# -------------------------------------------------------------
#  Combined Loss: MSE + SSIM-proxy (local variance penalty)
# -------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    """
    Weighted sum of MSE and a structural penalty.
    MSE alone ignores local structure; adding a spatial gradient
    term encourages the model to reproduce sharp star profiles.
    """

    def __init__(self, grad_weight: float = 0.1):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.grad_weight = grad_weight

    def _gradient_loss(self, original, recon):
        # Sobel-like finite differences
        dx_orig  = original[:, :, :, 1:] - original[:, :, :, :-1]
        dy_orig  = original[:, :, 1:, :] - original[:, :, :-1, :]
        dx_recon = recon[:, :, :, 1:]    - recon[:, :, :, :-1]
        dy_recon = recon[:, :, 1:, :]    - recon[:, :, :-1, :]
        return self.mse(dx_orig, dx_recon) + self.mse(dy_orig, dy_recon)

    def forward(self, original, recon):
        loss_mse  = self.mse(recon, original)
        loss_grad = self._gradient_loss(original, recon)
        return loss_mse + self.grad_weight * loss_grad, loss_mse.item(), loss_grad.item()


# -------------------------------------------------------------
#  Training Loop
# -------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_mse  = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:>3d} [Train]", leave=False)
    for tiles, _ in pbar:
        tiles = tiles.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        recon, _ = model(tiles)
        loss, mse_val, _ = criterion(tiles, recon)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse  += mse_val
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / n_batches, total_mse / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:>3d} [Val  ]", leave=False)
    for tiles, _ in pbar:
        tiles = tiles.to(device, non_blocking=True)
        recon, _ = model(tiles)
        loss, _, _ = criterion(tiles, recon)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


# -------------------------------------------------------------
#  Plot training curves
# -------------------------------------------------------------

def save_training_curve(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", color="#3B82F6", linewidth=2)
    ax.plot(val_losses,   label="Val Loss",   color="#F97316", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_title("Autoencoder Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Training curve saved -> {out_path}")


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    # -- CLI args ---------------------------------------------
    parser = argparse.ArgumentParser(description="Train Astro Autoencoder")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the best saved checkpoint")
    args = parser.parse_args()

    # -- Config -----------------------------------------------
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    cfg      = load_config(cfg_path)

    set_seed(42)
    device = get_device()

    # Resolve data paths via portable data_root
    train_dir, val_dir = get_data_dirs(cfg)

    print(f"\n{'='*60}")
    print("  Astro-Anomaly-Detector - Autoencoder Training")
    print(f"{'='*60}")
    print(f"  Train dir : {train_dir}")
    print(f"  Val dir   : {val_dir}")

    # -- Data -------------------------------------------------
    t_cfg   = cfg["training"]
    d_cfg   = cfg["data"]
    max_t   = None  # Use all tiles

    train_loader, val_loader = get_dataloaders(
        train_dir    = train_dir,
        val_dir      = val_dir,
        batch_size   = t_cfg["batch_size"],
        num_workers  = 0,          # Safe on Windows
        max_tiles    = max_t,
    )

    # -- Model ------------------------------------------------
    m_cfg = cfg["model"]
    model = AstroAutoencoder(
        n_channels   = d_cfg["num_channels"],
        base_channels= m_cfg["base_channels"],
        latent_dim   = m_cfg["latent_dim"],
    ).to(device)

    # -- Optimiser + Scheduler --------------------------------
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
    criterion = ReconstructionLoss(grad_weight=0.1)

    # -- Output dirs ------------------------------------------
    o_cfg       = cfg["output"]
    ckpt_dir    = os.path.join(PROJECT_ROOT, o_cfg["checkpoint_dir"])
    report_dir  = os.path.join(PROJECT_ROOT, o_cfg["report_dir"])
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    best_ckpt  = os.path.join(ckpt_dir, o_cfg["best_model_name"])
    last_ckpt  = os.path.join(ckpt_dir, "last_autoencoder.pth")
    curve_path = os.path.join(report_dir, "training_curve.png")

    # -- Resume from checkpoint? -------------------------------
    start_epoch = 1
    best_val    = float("inf")

    if args.resume and os.path.exists(best_ckpt):
        resumed_epoch, resumed_val = load_checkpoint(model, best_ckpt, device, optimizer)
        start_epoch = resumed_epoch + 1
        best_val    = resumed_val
        # Fast-forward the cosine scheduler to the correct position
        for _ in range(resumed_epoch):
            scheduler.step()
        print(f"  >>  Resuming from epoch {resumed_epoch}  (best val_loss = {resumed_val:.6f})")
        print(f"     Will run epochs {start_epoch} -> {t_cfg['epochs']}\n")
    elif args.resume:
        print(f"  [WARN] --resume requested but no checkpoint found at {best_ckpt}")
        print("         Starting fresh.\n")

    print(f"  Model parameters : {count_parameters(model):,}")
    print(f"  Latent dim       : {m_cfg['latent_dim']}")
    print(f"  Batch size       : {t_cfg['batch_size']}")
    print(f"  Epochs           : {start_epoch} -> {t_cfg['epochs']}")
    print(f"  Learning rate    : {t_cfg['learning_rate']}\n")

    # -- Training loop ----------------------------------------
    train_hist = []
    val_hist   = []
    t_start    = time.time()

    for epoch in range(start_epoch, t_cfg["epochs"] + 1):
        t_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        v_loss    = validate(model, val_loader, criterion, device, epoch)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        train_hist.append(t_loss)
        val_hist.append(v_loss)

        elapsed = (time.time() - t_start) / 60
        print(
            f"Epoch {epoch:>3d}/{t_cfg['epochs']}  "
            f"| Train: {t_loss:.5f}  "
            f"| Val: {v_loss:.5f}  "
            f"| LR: {lr_now:.2e}  "
            f"| {elapsed:.1f} min"
        )

        # Save best model
        if v_loss < best_val:
            best_val = v_loss
            save_checkpoint(model, optimizer, epoch, v_loss, best_ckpt)
            print(f"  [BEST] New best model saved  (val_loss = {v_loss:.6f})")

        # Periodic checkpoint
        if epoch % t_cfg["save_every"] == 0:
            periodic = os.path.join(ckpt_dir, f"autoencoder_epoch{epoch:03d}.pth")
            save_checkpoint(model, optimizer, epoch, v_loss, periodic)

    # Save last checkpoint
    save_checkpoint(model, optimizer, t_cfg["epochs"], v_loss, last_ckpt)

    # Save training curve
    save_training_curve(train_hist, val_hist, curve_path)

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  Training complete in {total_min:.1f} min")
    print(f"  Best Val Loss : {best_val:.6f}")
    print(f"  Best model    : {best_ckpt}")
    print(f"{'='*60}\n")
    print("Next step -> run: python src/detect_anomalies.py")


if __name__ == "__main__":
    main()
