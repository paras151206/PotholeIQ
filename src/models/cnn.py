"""
cnn.py — Custom CNN architecture for PotholeIQ (Module M3)

Architecture summary
--------------------
Input : (B, 3, 128, 128)

ConvBlock-1 : Conv(3->32,  3×3, pad=1) -> BN -> ReLU -> MaxPool(2)  -> (B,  32, 64, 64)
ConvBlock-2 : Conv(32->64, 3×3, pad=1) -> BN -> ReLU -> MaxPool(2)  -> (B,  64, 32, 32)
ConvBlock-3 : Conv(64->128,3×3, pad=1) -> BN -> ReLU -> MaxPool(2)  -> (B, 128, 16, 16)
ConvBlock-4 : Conv(128->256,3×3,pad=1) -> BN -> ReLU -> MaxPool(2)  -> (B, 256,  8,  8)

GlobalAvgPool : AdaptiveAvgPool2d(1)                               -> (B, 256,  1,  1)
Flatten                                                            -> (B, 256)

Classifier head:
  Linear(256->128) -> ReLU -> Dropout(p) -> Linear(128->2)            -> (B, 2)

Three variants are exposed:
  build_cnn()           — full model (BN + Dropout)
  build_cnn_no_bn()     — ablation: no BatchNorm
  build_cnn_no_dropout()— ablation: Dropout(0) ≡ identity
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.utils import CFG


# ---------------------------------------------------------------------------
# Building block: ConvBlock
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    One stage of the CNN feature extractor.

    Conv2d(in_ch, out_ch, 3×3, padding=1)  — 'same' spatial convolution
    BatchNorm2d(out_ch)                      — stabilises training (optional)
    ReLU(inplace=True)                       — element-wise non-linearity
    MaxPool2d(2, stride=2)                   — halves spatial dimensions

    Parameters
    ----------
    in_ch      : int   input channel count
    out_ch     : int   output channel count
    use_bn     : bool  whether to include BatchNorm (set False for ablation)
    """

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()

        layers = []

        # Spatial convolution — padding=1 keeps H×W unchanged before pooling
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
        # bias=False when BN follows because BN already shifts the activations

        if use_bn:
            # BatchNorm normalises per-channel; reduces internal covariate shift
            layers.append(nn.BatchNorm2d(out_ch))

        layers.append(nn.ReLU(inplace=True))

        # MaxPool halves spatial size: H × W -> H/2 × W/2
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Main model: CustomCNN
# ---------------------------------------------------------------------------

class CustomCNN(nn.Module):
    """
    Four-stage convolutional network for binary pothole classification.

    Parameters
    ----------
    channels   : list[int]  output channels for each ConvBlock (len == 4)
    dropout    : float      dropout probability in the classifier head
    num_classes: int        number of output logits (2)
    use_bn     : bool       include BatchNorm in every ConvBlock
    """

    def __init__(
        self,
        channels: list = None,
        dropout: float = None,
        num_classes: int = 2,
        use_bn: bool = True,
    ):
        super().__init__()

        if channels is None:
            channels = list(CFG.cnn.channels)   # [32, 64, 128, 256]
        if dropout is None:
            dropout = float(CFG.cnn.dropout)    # 0.4

        # --- Feature extractor -------------------------------------------
        # Build 4 ConvBlocks; each halves spatial dimensions and increases depth.
        # Starting spatial: 128×128
        # After block-1: 64×64  |  After block-2: 32×32
        # After block-3: 16×16  |  After block-4:  8×8
        in_ch = 3  # RGB input
        conv_blocks = []
        for out_ch in channels:
            conv_blocks.append(ConvBlock(in_ch, out_ch, use_bn=use_bn))
            in_ch = out_ch

        self.features = nn.Sequential(*conv_blocks)
        # Output: (B, 256, 8, 8)

        # --- Global Average Pooling ---------------------------------------
        # Collapses 8×8 spatial maps to a single value per channel.
        # Significantly reduces parameter count vs a Flatten->Dense approach.
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Output: (B, 256, 1, 1)

        # --- Classifier head ---------------------------------------------
        head_in = channels[-1]   # 256
        self.classifier = nn.Sequential(
            nn.Flatten(),                              # (B, 256)
            nn.Linear(head_in, 128),                  # (B, 128)
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),                    # stochastic regularisation
            nn.Linear(128, num_classes),              # (B, 2)
        )

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """
        Kaiming (He) for Conv2d layers (ReLU-compatible).
        Xavier for Linear layers.
        BatchNorm layers start at weight=1, bias=0 (PyTorch default).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 128, 128)  normalised image tensor

        Returns
        -------
        (B, 2)  logits for [normal, pothole]
        """
        x = self.features(x)   # (B, 256, 8, 8)
        x = self.gap(x)        # (B, 256, 1, 1)
        x = self.classifier(x) # (B, 2)
        return x

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base = super().__repr__()
        return f"{base}\nTotal params: {total_params:,}  Trainable: {trainable:,}"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_cnn() -> CustomCNN:
    """
    Full CustomCNN with BatchNorm and Dropout as configured in CFG.cnn.

    Returns
    -------
    CustomCNN  — ready for training
    """
    return CustomCNN(
        channels=list(CFG.cnn.channels),
        dropout=float(CFG.cnn.dropout),
        num_classes=2,
        use_bn=True,
    )


def build_cnn_no_bn() -> CustomCNN:
    """
    Ablation variant: CustomCNN **without** BatchNorm.

    Use to quantify how much BN contributes to convergence speed and accuracy.
    All other hyper-parameters remain identical.

    Returns
    -------
    CustomCNN  — with use_bn=False
    """
    return CustomCNN(
        channels=list(CFG.cnn.channels),
        dropout=float(CFG.cnn.dropout),
        num_classes=2,
        use_bn=False,
    )


def build_cnn_no_dropout() -> CustomCNN:
    """
    Ablation variant: CustomCNN **without** Dropout (p=0).

    Use to quantify regularisation contribution from Dropout.
    Setting p=0 means Dropout is a no-op at both train and eval time.

    Returns
    -------
    CustomCNN  — with dropout=0.0
    """
    return CustomCNN(
        channels=list(CFG.cnn.channels),
        dropout=0.0,
        num_classes=2,
        use_bn=True,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name, fn in [
        ("Full CNN", build_cnn),
        ("No BN",    build_cnn_no_bn),
        ("No Drop",  build_cnn_no_dropout),
    ]:
        m = fn()
        dummy = torch.randn(4, 3, 128, 128)
        out = m(dummy)
        assert out.shape == (4, 2), f"[{name}] Bad output shape: {out.shape}"
        total = sum(p.numel() for p in m.parameters())
        print(f"{name}: input {tuple(dummy.shape)} -> output {tuple(out.shape)}  |  params={total:,}")
