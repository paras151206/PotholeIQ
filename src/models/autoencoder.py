"""
autoencoder.py — Convolutional Autoencoder for anomaly detection (Module M8)

Architecture
------------
The autoencoder is trained ONLY on normal (non-pothole) road images.
At inference time, normal images reconstruct well (low MSE); potholes
reconstruct poorly (high MSE) because the model has never seen them.
This MSE score is the "anomaly score".

Encoder: 4 strided convolutions → bottleneck → latent vector (64-dim)
Decoder: 4 transposed convolutions → reconstructed image

Input/Output: (B, 3, 128, 128)  float32  normalised to ImageNet stats

Latent dim: 64 (from CFG.autoencoder.latent_dim)
Spatial resolution at bottleneck: 8×8 → flattened to 256*8*8 = 16384
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.utils import CFG


# ─────────────────────────── Encoder ─────────────────────────────────────────

class Encoder(nn.Module):
    """
    Encodes (B, 3, 128, 128) → latent vector (B, latent_dim).

    Each Conv layer halves the spatial resolution (stride=2):
      128 → 64 → 32 → 16 → 8
    Channel progression: 3 → 32 → 64 → 128 → 256
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim  = latent_dim
        self._spatial    = 8          # spatial size after 4 stride-2 convolutions
        self._ch         = 256        # channels at bottleneck
        self.flat_dim    = self._ch * self._spatial * self._spatial  # 256*8*8 = 16384

        self.conv = nn.Sequential(
            # (B,  3, 128, 128) → (B, 32, 64, 64)
            nn.Conv2d(3,   32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            # (B, 32,  64,  64) → (B, 64, 32, 32)
            nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            # (B, 64,  32,  32) → (B, 128,16, 16)
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            # (B, 128, 16,  16) → (B, 256, 8,  8)
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        # Flatten + project to latent dim
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                        # (B, 256, 8, 8)
        x = x.view(x.size(0), -1)              # (B, 16384)
        z = self.fc(x)                          # (B, latent_dim)
        return z


# ─────────────────────────── Decoder ─────────────────────────────────────────

class Decoder(nn.Module):
    """
    Decodes latent vector (B, latent_dim) → (B, 3, 128, 128).

    Mirror image of Encoder: linear projection → reshape → 4 transposed convolutions.
    Tanh activation at output maps to [-1, 1], matching normalised input range.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self._spatial = 8
        self._ch      = 256
        self.flat_dim = self._ch * self._spatial * self._spatial

        self.fc = nn.Linear(latent_dim, self.flat_dim)

        self.deconv = nn.Sequential(
            # (B, 256, 8, 8) → (B, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            # (B, 128, 16, 16) → (B, 64, 32, 32)
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            # (B, 64,  32, 32) → (B, 32, 64, 64)
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            # (B, 32,  64, 64) → (B, 3, 128, 128)
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                                                      # (B, flat_dim)
        x = x.view(x.size(0), self._ch, self._spatial, self._spatial)      # (B, 256, 8, 8)
        x = self.deconv(x)                                                  # (B, 3, 128, 128)
        return x


# ─────────────────────────── Autoencoder ─────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    Full convolutional autoencoder combining Encoder + Decoder.

    Forward pass returns the reconstructed image (same shape as input).
    Use compute_anomaly_score() for inference on a single image.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z    = self.encoder(x)   # compress
        x_hat = self.decoder(z)  # reconstruct
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation for a batch."""
        return self.encoder(x)


def build_autoencoder() -> ConvAutoencoder:
    """Factory: build an autoencoder using latent_dim from config."""
    return ConvAutoencoder(latent_dim=CFG.autoencoder.latent_dim)


# ─────────────────────────── Inference helper ─────────────────────────────────

def compute_anomaly_score(
    model: ConvAutoencoder,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Compute MSE reconstruction error for a single image tensor.

    Parameters
    ----------
    model      : trained ConvAutoencoder
    img_tensor : shape (3, H, W) — normalised, NOT batched
    device     : torch device

    Returns
    -------
    float — MSE between input and reconstruction (higher = more anomalous)
    """
    model.eval()
    with torch.no_grad():
        x     = img_tensor.unsqueeze(0).to(device)   # (1, 3, H, W)
        x_hat = model(x)
        score = torch.mean((x - x_hat) ** 2).item()
    return score
