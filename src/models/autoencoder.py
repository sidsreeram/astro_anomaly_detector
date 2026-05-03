"""
Convolutional Autoencoder for Astronomical Anomaly Detection
=============================================================
Architecture:
  - Encoder: 4 strided-conv blocks   (128 → 64 → 32 → 16 → 8)
  - Bottleneck: Flat dense latent code  (z = 256-dim)
  - Decoder: 4 transposed-conv blocks  (8 → 16 → 32 → 64 → 128)

The key insight for anomaly detection:
  Normal tiles (uniform star fields) are easily reconstructable.
  Anomalies (cosmic rays, blended objects, sensor artifacts) produce
  HIGH reconstruction error → high anomaly score.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
#  Building Blocks
# ─────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Conv → BatchNorm → LeakyReLU (stride=2 for downsampling)."""

    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """ConvTranspose → BatchNorm → ReLU (upsampling)."""

    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1, last=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        ]
        if last:
            layers.append(nn.Sigmoid())          # Output in [0, 1] to match normalised input
        else:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────
#  Full Autoencoder
# ─────────────────────────────────────────────────────────────

class AstroAutoencoder(nn.Module):
    """
    Symmetric Convolutional Autoencoder.

    Input:  (B, 4, 128, 128)   — 4-channel normalised tile
    Output: (B, 4, 128, 128)   — reconstructed tile  (same shape)

    Intermediate spatial sizes (with base_channels=32):
      128 → 64 → 32 → 16 → 8   [encoder]
                               → flat → latent_dim → flat → flatten back
       8 → 16 → 32 → 64 → 128  [decoder]
    """

    def __init__(self, n_channels: int = 4, base_channels: int = 32, latent_dim: int = 256):
        super().__init__()
        b = base_channels  # shorthand

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = EncoderBlock(n_channels, b)        # 128 → 64
        self.enc2 = EncoderBlock(b,       b * 2)       # 64  → 32
        self.enc3 = EncoderBlock(b * 2,   b * 4)       # 32  → 16
        self.enc4 = EncoderBlock(b * 4,   b * 8)       # 16  → 8

        # Feature map size after enc4: (B, b*8, 8, 8)
        self._flat_size = b * 8 * 8 * 8               # = 32*8 * 64 = 16384

        # ── Bottleneck ────────────────────────────────────────
        self.fc_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, self._flat_size),
            nn.ReLU(inplace=True),
        )
        self._reshape = (b * 8, 8, 8)

        # ── Decoder ──────────────────────────────────────────
        self.dec1 = DecoderBlock(b * 8, b * 4)        # 8  → 16
        self.dec2 = DecoderBlock(b * 4, b * 2)        # 16 → 32
        self.dec3 = DecoderBlock(b * 2, b)            # 32 → 64
        self.dec4 = DecoderBlock(b,     n_channels, last=True)  # 64 → 128

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_dec(z)
        x = x.view(-1, *self._reshape)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ─────────────────────────────────────────────────────────────
#  Anomaly Scoring Utilities
# ─────────────────────────────────────────────────────────────

def reconstruction_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE loss (not averaged across batch)."""
    diff = (original - reconstructed) ** 2
    # Mean over C, H, W — keep B dimension
    return diff.mean(dim=(1, 2, 3))


def reconstruction_error_map(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Pixel-level squared error map averaged across channels.
    Shape: (B, H, W)  — useful for spatial anomaly heatmaps.
    """
    diff = (original - reconstructed) ** 2
    return diff.mean(dim=1)   # average across channels


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = AstroAutoencoder(n_channels=4, base_channels=32, latent_dim=256)
    x = torch.randn(2, 4, 128, 128)
    recon, z = model(x)
    print(f"Input  : {x.shape}")
    print(f"Latent : {z.shape}")
    print(f"Output : {recon.shape}")
    print(f"Params : {count_parameters(model):,}")
    mse = reconstruction_mse(x, recon)
    print(f"MSE scores per sample: {mse.detach().numpy()}")
