import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vae import ResNetVAE
from src.utils.helpers import load_config, get_device, load_checkpoint

def plot_saliency_heatmap(model, x, out_path, target_loss='mse'):
    """
    x: A single input tensor of shape (1, 4, 128, 128).
    Creates a pixel-level heatmap defining why a tile is anomalous.
    """
    model.eval()
    x.requires_grad_()
    
    recon_x, mu, logvar = model(x)
    
    # We want to see what pixels drive the MSE error up
    loss = F.mse_loss(recon_x, x, reduction='sum')
    model.zero_grad()
    loss.backward()
    
    # Calculate Saliency map
    # Take absolute gradients, sum them across the 4 channels to get an aggregated (128x128) heatmap
    saliency = x.grad.abs().squeeze(0).sum(dim=0)
    
    # Normalize for visualization
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency_np = saliency.cpu().numpy()
    
    # Build a standard RGB false-color representation of the first 3 channels
    rgb_input = x.squeeze(0)[:3].detach().cpu().permute(1, 2, 0).numpy()
    rgb_input = np.clip(rgb_input, 0, 1)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#0D1117")
    
    titles = ["Original (Ch0-2)", "Reconstructed (Ch0-2)", "Error Gradient Saliency Map", "XAI Image Overlay"]
    
    # Original
    axes[0].imshow(rgb_input)
    
    # Recon
    rgb_recon = recon_x.squeeze(0)[:3].detach().cpu().permute(1, 2, 0).numpy()
    rgb_recon = np.clip(rgb_recon, 0, 1)
    axes[1].imshow(rgb_recon)

    # Saliency
    im = axes[2].imshow(saliency_np, cmap='inferno')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[3].imshow(rgb_input)
    axes[3].imshow(saliency_np, cmap='inferno', alpha=0.5)
    
    for ax, title in zip(axes, titles):
        ax.set_title(title, color="white", fontsize=12)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return saliency_np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", type=str, required=True, help="Path to a .npy tile to explain")
    args = parser.parse_args()
    
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    device = get_device()
    
    m_cfg = cfg["model"]
    model = ResNetVAE(
        in_channels=cfg["data"]["num_channels"],
        base_channels=m_cfg["base_channels"],
        latent_dim=m_cfg["latent_dim"]
    ).to(device)
    
    ckpt_path = os.path.join(PROJECT_ROOT, cfg["output"]["checkpoint_dir"], "best_vae.pth")
    if os.path.exists(ckpt_path):
         load_checkpoint(model, ckpt_path, device)
    else:
         print(f"[Warning] No checkpoint at {ckpt_path}. Using untrained model (meaningless saliency).")
         
    tile_np = np.load(args.tile)
    if tile_np.shape != (4, 128, 128):
         print(f"Error: Expected shape (4,128,128), got {tile_np.shape}")
         sys.exit(1)
         
    x_tensor = torch.from_numpy(tile_np).unsqueeze(0).to(device)
    out_name = os.path.basename(args.tile).replace(".npy", "_xai.png")
    out_path = os.path.join(PROJECT_ROOT, cfg["output"]["report_dir"], out_name)
    
    plot_saliency_heatmap(model, x_tensor, out_path)
    print(f"Saved XAI saliency map to {out_path}")
