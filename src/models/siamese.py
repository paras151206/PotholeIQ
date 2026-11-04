"""
src/models/siamese.py
======================
Siamese network with contrastive loss for pothole similarity learning.

The network shares a single encoder across both input branches (weight sharing
is enforced by using the same nn.Module instance for both forward passes).
Embeddings are L2-normalised so distances live on a unit hypersphere, which
makes the contrastive margin directly interpretable.

NOTE: Positive pairs in this project are augmented views of the *same* image,
NOT real same-pothole-over-time pairs.  This is a limitation documented in
README.md.  The model therefore learns "is this pair from the same damage
category?" rather than "has this specific pothole changed over time?".
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import CFG
from src.models.cnn import build_cnn   # CustomCNN for the shared backbone


# ---------------------------------------------------------------------------
# SiameseNet
# ---------------------------------------------------------------------------
class SiameseNet(nn.Module):
    """Siamese network with a shared CustomCNN backbone.

    The backbone is the conv_blocks + GAP portion of CustomCNN (no classifier).
    A linear projection then maps from the backbone feature dimension (256) to
    the configured embedding_dim (default 128), followed by L2 normalisation.

    L2 normalisation ensures that all embeddings lie on a unit hypersphere,
    which simplifies the interpretation of Euclidean distances and makes
    contrastive training more stable.
    """

    def __init__(self, embedding_dim: int = None):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = CFG.siamese.embedding_dim   # 128

        # Build a full CustomCNN and strip the classifier to get just the
        # feature extractor.  Shared parameters → identical weights for
        # both branches by construction (a single Module instance).
        cnn = build_cnn()
        # CNN attributes: .features (Sequential of ConvBlocks), .gap, .classifier
        self.feature_blocks = cnn.features   # Sequential of ConvBlocks
        self.gap_layer       = cnn.gap       # AdaptiveAvgPool2d(1)

        # Determine backbone output dim by a dry run
        with torch.no_grad():
            dummy        = torch.zeros(1, 3, 128, 128)
            feat         = self._extract_features(dummy)
            backbone_dim = feat.shape[1]  # 256

        self.embedding_layer = nn.Linear(backbone_dim, embedding_dim)

    # --- Internal helper ---------------------------------------------------- #

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run x through conv blocks → GAP → flatten.

        Returns:
            (B, backbone_dim) flattened feature vector.
        """
        x = self.feature_blocks(x)   # (B, 256, 8, 8)
        x = self.gap_layer(x)        # (B, 256, 1, 1)
        x = torch.flatten(x, 1)      # (B, 256)
        return x

    # --- Public API --------------------------------------------------------- #

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Extract an L2-normalised embedding for a single-branch input.

        Args:
            x: (B, 3, H, W) image tensor

        Returns:
            (B, embedding_dim) L2-normalised embedding on the unit hypersphere
        """
        feat = self._extract_features(x)       # (B, backbone_dim)
        emb  = self.embedding_layer(feat)      # (B, embedding_dim)
        emb  = F.normalize(emb, p=2, dim=1)   # L2 norm → unit sphere
        return emb

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """Forward both branches with shared weights.

        Args:
            x1, x2: (B, 3, H, W) image pair

        Returns:
            (emb1, emb2): both (B, embedding_dim) L2-normalised embeddings
        """
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------
class ContrastiveLoss(nn.Module):
    """Contrastive loss (Hadsell et al., 2006).

    Equation:
        d   = ||emb1 - emb2||_2                       (Euclidean distance)
        L   = (1 - y) * d²  +  y * max(0, m - d)²

    where y=0 → same class (pull together), y=1 → different class (push apart
    if closer than margin m).

    The margin *m* defines the minimum distance enforced between embeddings of
    different classes.  Pairs already further apart than m contribute 0 loss.
    """

    def __init__(self, margin: float = None):
        super().__init__()
        self.margin = margin if margin is not None else CFG.siamese.margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            emb1, emb2: (B, D) L2-normalised embeddings
            label:      (B,) int tensor — 0=same class, 1=different class

        Returns:
            Scalar mean contrastive loss.
        """
        # Euclidean distance between embedding pairs
        # d = ||emb1 - emb2||_2  for each sample in the batch
        d = F.pairwise_distance(emb1, emb2, p=2)   # (B,)

        label = label.float()

        # Same-class term: (1-y) * d²  — penalises large distances for same pairs
        same_loss = (1.0 - label) * d.pow(2)

        # Different-class term: y * max(0, m - d)²  — penalises pairs that are
        # too close (closer than the margin m)
        diff_loss = label * torch.clamp(self.margin - d, min=0.0).pow(2)

        loss = 0.5 * (same_loss + diff_loss).mean()
        return loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_siamese() -> SiameseNet:
    """Instantiate SiameseNet using parameters from config.yaml."""
    return SiameseNet(embedding_dim=CFG.siamese.embedding_dim)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net  = build_siamese()
    x1   = torch.randn(4, 3, 128, 128)
    x2   = torch.randn(4, 3, 128, 128)
    e1, e2 = net(x1, x2)
    print(f"Embedding shape: {e1.shape}")            # (4, 128)
    print(f"L2 norms (should be ~1): {e1.norm(dim=1)}")

    criterion = ContrastiveLoss()
    labels = torch.tensor([0, 1, 0, 1])
    loss = criterion(e1, e2, labels)
    print(f"Contrastive loss: {loss.item():.4f}")
    print("siamese.py OK")
