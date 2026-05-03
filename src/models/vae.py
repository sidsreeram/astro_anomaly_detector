import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetVAE(nn.Module):
    """
    ResNet-based Variational Autoencoder.
    Input:  (B, in_channels, 128, 128)
    Output: Reconstructed (B, in_channels, 128, 128), mu, logvar
    """
    def __init__(self, in_channels=4, base_channels=32, latent_dim=256):
        super().__init__()
        
        # Encoder (128x128 -> 8x8)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            ResBlock(base_channels, base_channels*2, stride=2),   # -> 64x64
            ResBlock(base_channels*2, base_channels*4, stride=2), # -> 32x32
            ResBlock(base_channels*4, base_channels*8, stride=2), # -> 16x16
            ResBlock(base_channels*8, base_channels*8, stride=2), # -> 8x8
        )
        
        self.flat_size = base_channels * 8 * 8 * 8
        self.reshape_dim = base_channels * 8
        
        # VAE Latent space projections
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        
        # We use standard Convs and upsampling to avoid checkerboard artifacts often caused by ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # -> 16x16
            ResBlock(base_channels*8, base_channels*4, stride=1),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # -> 32x32
            ResBlock(base_channels*4, base_channels*2, stride=1),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # -> 64x64
            ResBlock(base_channels*2, base_channels, stride=1),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # -> 128x128
            ResBlock(base_channels, base_channels, stride=1),
            
            nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # Maintain [0, 1] output bounds
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.reshape_dim, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reconstruction_error_map(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Pixel-level squared error map averaged across channels.
    Shape: (B, H, W)  — useful for spatial anomaly heatmaps.
    """
    diff = (original - reconstructed) ** 2
    return diff.mean(dim=1)   # average across channels
