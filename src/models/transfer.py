"""
transfer.py — MobileNetV2 transfer learning model for PotholeIQ (Module M4)

Design rationale
----------------
MobileNetV2 was pre-trained on ImageNet (1.28 M images, 1000 classes).
Its learned low-level filters (edges, textures, colours) transfer well to
road-surface images. We replace only the final classification head and
train in two phases:

  Phase 1 — Feature extraction
      Backbone frozen; only the new head is updated.
      High LR is safe because the backbone weights are protected.

  Phase 2 — Fine-tuning
      Last `n` InvertedResidual blocks of the backbone are unfrozen.
      Much lower LR to avoid catastrophic forgetting of ImageNet features.

MobileNetV2 feature architecture (torchvision):
  model.features   — Sequential of 19 blocks (index 0–18)
    [0]              : ConvBNReLU (stem)
    [1–18]           : InvertedResidual blocks
  model.classifier — Linear(1280, 1000)   ← we replace this

New classifier head:
  Dropout(dropout)         → reduces overfitting in the head
  Linear(1280, 256)        → compresses to a task-specific representation
  ReLU()
  Dropout(dropout / 2)     → lighter regularisation before logits
  Linear(256, 2)           → binary classification logits
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torchvision import models

from src.utils import CFG


# ---------------------------------------------------------------------------
# TransferModel
# ---------------------------------------------------------------------------

class TransferModel(nn.Module):
    """
    MobileNetV2 with a custom two-layer classifier head.

    Parameters
    ----------
    dropout    : float   dropout probability (applied before the first Linear)
    num_classes: int     number of output logits (2)
    pretrained : bool    load ImageNet weights if True
    """

    def __init__(
        self,
        dropout: float = None,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        if dropout is None:
            dropout = float(CFG.transfer.dropout)   # 0.3

        # -----------------------------------------------------------------
        # Backbone: MobileNetV2
        # -----------------------------------------------------------------
        # torchvision ≥ 0.13 uses `weights=` kwarg; older versions use `pretrained=`.
        # We try the new API and fall back gracefully.
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
        except AttributeError:
            # torchvision < 0.13 compatibility
            backbone = models.mobilenet_v2(pretrained=pretrained)

        # Retain the feature extractor; discard the original 1000-class head
        self.features = backbone.features
        # Output of features: (B, 1280, 4, 4) for 128×128 input after GAP → (B, 1280)

        # MobileNetV2 has a built-in AdaptiveAvgPool2d(1) in the classifier path.
        # We replicate it here so our custom head receives a flat (B, 1280) vector.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # -----------------------------------------------------------------
        # Custom classification head
        # -----------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),                   # (B, 1280)
            nn.Linear(1280, 256),                    # (B, 256)
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),               # lighter regularisation
            nn.Linear(256, num_classes),             # (B, 2)
        )

        # Initialise only the new head; backbone keeps ImageNet weights
        self._init_head()

    # ------------------------------------------------------------------
    def _init_head(self):
        """Xavier initialisation for the Linear layers in our new head."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """
        Freeze ALL backbone (features) parameters.

        Called at the start of Phase 1 so only the classifier head receives
        gradient updates. This is safe to call multiple times.
        """
        for param in self.features.parameters():
            param.requires_grad = False
        print("[TransferModel] Backbone frozen.")

    # ------------------------------------------------------------------
    def unfreeze_last_blocks(self, n: int = 3):
        """
        Unfreeze the last `n` InvertedResidual blocks (and all subsequent layers).

        MobileNetV2 has 19 blocks in self.features (indices 0–18).
        Unfreezing from index (19 - n) onwards lets the backbone adapt
        its high-level features to road-surface textures while preserving
        the low-level edge/texture filters learned on ImageNet.

        Parameters
        ----------
        n : int   number of trailing feature blocks to unfreeze (default 3)
        """
        total_blocks = len(self.features)             # 19
        unfreeze_from = max(0, total_blocks - n)      # e.g. 16 when n=3

        for idx, block in enumerate(self.features):
            should_unfreeze = idx >= unfreeze_from
            for param in block.parameters():
                param.requires_grad = should_unfreeze

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[TransferModel] Unfroze last {n} feature block(s) "
            f"(indices {unfreeze_from}–{total_blocks-1}).  "
            f"Trainable params: {trainable:,}"
        )

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
        x = self.features(x)        # (B, 1280, H', W')   H'=W'=4 for 128px input
        x = self.pool(x)            # (B, 1280, 1, 1)
        x = x.flatten(1)            # (B, 1280)
        x = self.classifier(x)      # (B, 2)
        return x

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        base = super().__repr__()
        return (
            f"{base}\n"
            f"Total params: {total:,}  |  "
            f"Trainable: {trainable:,}  |  Frozen: {frozen:,}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_transfer_model(pretrained: bool = True) -> TransferModel:
    """
    Construct a TransferModel using hyper-parameters from CFG.

    Parameters
    ----------
    pretrained : bool   whether to load ImageNet weights (default True)

    Returns
    -------
    TransferModel  — backbone NOT frozen yet (call freeze_backbone() externally)
    """
    model = TransferModel(
        dropout=float(CFG.transfer.dropout),
        num_classes=2,
        pretrained=pretrained,
    )
    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_transfer_model(pretrained=False)   # skip download in quick test

    # Phase 1 setup
    model.freeze_backbone()
    frozen_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 1 trainable params: {frozen_p1:,}")

    # Phase 2 setup
    model.unfreeze_last_blocks(n=3)
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2 trainable params: {trainable_p2:,}")

    # Forward-pass check
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    assert out.shape == (2, 2), f"Unexpected output shape: {out.shape}"
    print(f"\nForward pass: {tuple(dummy.shape)} → {tuple(out.shape)}  ✓")
