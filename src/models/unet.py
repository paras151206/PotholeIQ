"""
src/models/unet.py
==================
Lightweight U-Net for binary road-pothole segmentation.

Architecture (input: B×3×128×128):
  Encoder:
    enc1: DoubleConv(3,   32)   → (B, 32,  128,128)
    pool → (B, 32,  64, 64)
    enc2: DoubleConv(32,  64)   → (B, 64,  64, 64)
    pool → (B, 64,  32, 32)
    enc3: DoubleConv(64, 128)   → (B,128,  32, 32)
    pool → (B,128,  16, 16)
  Bottleneck:
    DoubleConv(128, 256)        → (B,256,  16, 16)
  Decoder:
    up3: ConvTranspose2d(256→128) → (B,128,32,32) ; cat enc3 → (B,256,32,32)
         DoubleConv(256,128)    → (B,128, 32, 32)
    up2: ConvTranspose2d(128→64)  → (B, 64,64,64) ; cat enc2 → (B,128,64,64)
         DoubleConv(128, 64)   → (B, 64, 64, 64)
    up1: ConvTranspose2d(64→32)   → (B, 32,128,128); cat enc1 → (B, 64,128,128)
         DoubleConv(64, 32)    → (B, 32,128,128)
  Output:
    Conv2d(32, 1, kernel=1) → Sigmoid → (B, 1,128,128)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from src.utils import CFG


# ---------------------------------------------------------------------------
# Building block: two consecutive Conv→BN→ReLU layers
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two (Conv2d → BatchNorm2d → ReLU) layers back-to-back.

    Using BN keeps gradients stable during training and reduces the need
    for careful weight-initialisation. Padding=1 preserves spatial size.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# SmallUNet
# ---------------------------------------------------------------------------
class SmallUNet(nn.Module):
    """Lightweight U-Net for 128×128 binary segmentation.

    encoder_channels = [3, 32, 64, 128]  (from config)
    The bottleneck doubles the last encoder channel → 256.
    Skip connections concatenate encoder feature maps into the decoder,
    which is the core idea of U-Net (Ronneberger et al., 2015).
    """

    def __init__(self, encoder_channels=None):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = CFG.unet.encoder_channels  # [3, 32, 64, 128]

        c0, c1, c2, c3 = encoder_channels          # 3, 32, 64, 128
        c4 = c3 * 2                                 # bottleneck: 256

        # ---- Encoder -------------------------------------------------------
        self.enc1 = DoubleConv(c0, c1)              # → (B, 32, 128, 128)
        self.enc2 = DoubleConv(c1, c2)              # → (B, 64,  64,  64)
        self.enc3 = DoubleConv(c2, c3)              # → (B,128,  32,  32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Bottleneck ----------------------------------------------------
        self.bottleneck = DoubleConv(c3, c4)        # → (B,256,  16,  16)

        # ---- Decoder -------------------------------------------------------
        # Each stage: upsample (doubles H,W), cat skip → DoubleConv
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)   # → (B,128,32,32)
        self.dec3 = DoubleConv(c3 + c3, c3)        # cat enc3: 128+128=256→128

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)   # → (B,64,64,64)
        self.dec2 = DoubleConv(c2 + c2, c2)        # cat enc2:  64+ 64=128→ 64

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)   # → (B,32,128,128)
        self.dec1 = DoubleConv(c1 + c1, c1)        # cat enc1:  32+ 32= 64→ 32

        # ---- Output head ---------------------------------------------------
        # 1×1 conv collapses channel dim to 1; sigmoid maps to [0,1] prob
        self.out_conv = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128) normalised input image

        Returns:
            (B, 1, 128, 128) segmentation probability map in [0, 1]
        """
        # Encoder – save feature maps for skip connections
        e1 = self.enc1(x)               # (B, 32,128,128)
        e2 = self.enc2(self.pool(e1))   # (B, 64, 64, 64)
        e3 = self.enc3(self.pool(e2))   # (B,128, 32, 32)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B,256, 16, 16)

        # Decoder – concatenate skip connections before DoubleConv
        d3 = self.up3(b)                        # (B,128, 32, 32)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # (B,128, 32, 32)

        d2 = self.up2(d3)                       # (B, 64, 64, 64)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 64, 64, 64)

        d1 = self.up1(d2)                       # (B, 32,128,128)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, 32,128,128)

        return self.out_conv(d1)                # (B,  1,128,128)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_unet() -> SmallUNet:
    """Instantiate SmallUNet using parameters from config.yaml."""
    return SmallUNet(encoder_channels=CFG.unet.encoder_channels)


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------
def compute_severity(mask_tensor: torch.Tensor, thresholds=None) -> str:
    """Classify pothole severity based on the fraction of pixels predicted as
    pothole in a binary segmentation mask.

    Args:
        mask_tensor: (1, H, W) or (H, W) float tensor with values in [0, 1].
                     Pixels > 0.5 are considered pothole.
        thresholds:  object with .low_max and .medium_max float attributes.
                     If None, reads from CFG.unet.severity_thresholds.

    Returns:
        'Low'    if area_ratio <= low_max    (default ≤ 0.05)
        'Medium' if area_ratio <= medium_max (default ≤ 0.20)
        'High'   otherwise
    """
    if thresholds is None:
        thresholds = CFG.unet.severity_thresholds

    # Flatten and compute ratio of pothole pixels
    binary = (mask_tensor > 0.5).float()
    area_ratio = binary.mean().item()

    if area_ratio <= thresholds.low_max:
        return "Low"
    elif area_ratio <= thresholds.medium_max:
        return "Medium"
    else:
        return "High"


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_unet()
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")          # Expected: (2, 1, 128, 128)
    print(f"Value range: [{out.min():.4f}, {out.max():.4f}]")

    # Test severity
    high_mask = torch.ones(1, 128, 128) * 0.9
    low_mask  = torch.ones(1, 128, 128) * 0.03
    print(f"Severity (high mask): {compute_severity(high_mask)}")
    print(f"Severity (low  mask): {compute_severity(low_mask)}")
    print("unet.py OK")
