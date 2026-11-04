"""
baseline.py — TinyMLP baseline model for PotholeIQ (Module M2)

Architecture overview:
    Input: flattened image tensor of shape (B, 49152)
           where 49152 = 3 × 128 × 128
    
    Flatten layer  → (B, 49152)          [handled inside forward()]
    Linear 1       → (B, 512)            first hidden layer
    ReLU + Dropout → (B, 512)            non-linearity + regularisation
    Linear 2       → (B, 128)            second hidden layer
    ReLU + Dropout → (B, 128)            non-linearity + regularisation
    Linear 3 (out) → (B, 2)             logits for [normal, pothole]

The MLP is intentionally tiny so it sets a low-but-honest performance floor.
Any CNN module that cannot beat this baseline indicates a training bug.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.utils import CFG


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    """
    A minimal multi-layer perceptron used as the M2 baseline.

    Parameters
    ----------
    input_dim : int
        Flattened image size. Default = 3 × 128 × 128 = 49_152.
    hidden_dims : list[int]
        Sizes of the two hidden layers. Read from CFG.baseline.hidden_dims.
    num_classes : int
        Number of output logits (2 for binary pothole detection).
    dropout_p : float
        Dropout probability applied after each hidden ReLU (default 0.3).
    """

    def __init__(
        self,
        input_dim: int = 3 * 128 * 128,
        hidden_dims: list = None,
        num_classes: int = 2,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = list(CFG.baseline.hidden_dims)  # [512, 128]

        # --- Layer 1: input → first hidden -----------------------------------
        # Projects the high-dimensional pixel vector down to 512 features.
        # Using a large first layer helps capture low-level pixel statistics.
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])

        # Non-linearity after layer 1
        self.relu1 = nn.ReLU(inplace=True)

        # Dropout prevents co-adaptation of neurons (regularisation)
        self.drop1 = nn.Dropout(p=dropout_p)

        # --- Layer 2: first hidden → second hidden ---------------------------
        # Compresses 512 → 128; forces the network to learn a compact
        # representation before the classification head.
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=dropout_p)

        # --- Output layer: second hidden → logits ----------------------------
        # Produces raw (un-normalised) scores for each class.
        # CrossEntropyLoss applies softmax internally, so no softmax here.
        self.out = nn.Linear(hidden_dims[1], num_classes)

        # Weight initialisation: He-uniform for ReLU layers
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Kaiming (He) initialisation for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, 128, 128)
            Raw image tensor as returned by PotholeDataset.

        Returns
        -------
        torch.Tensor, shape (B, 2)
            Class logits [normal_score, pothole_score].
        """
        # Flatten the spatial + channel dimensions into a 1-D vector per image.
        # (B, 3, 128, 128) → (B, 49152)
        x = x.view(x.size(0), -1)

        # Hidden layer 1
        x = self.drop1(self.relu1(self.fc1(x)))   # (B, 512)

        # Hidden layer 2
        x = self.drop2(self.relu2(self.fc2(x)))   # (B, 128)

        # Output logits
        x = self.out(x)                            # (B, 2)
        return x

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base = super().__repr__()
        return f"{base}\nTotal params: {total_params:,}  Trainable: {trainable:,}"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_baseline() -> TinyMLP:
    """
    Construct and return a TinyMLP using hyper-parameters from CFG.

    Returns
    -------
    TinyMLP
        Freshly initialised baseline model.
    """
    model = TinyMLP(
        input_dim=3 * CFG.data.image_size[0] * CFG.data.image_size[1],
        hidden_dims=list(CFG.baseline.hidden_dims),
        num_classes=2,
        dropout_p=0.3,
    )
    return model


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_baseline()
    print(model)

    # Feed a random batch and confirm output shape
    dummy = torch.randn(4, 3, 128, 128)
    out = model(dummy)
    assert out.shape == (4, 2), f"Unexpected output shape: {out.shape}"
    print(f"\nSanity check passed: input {tuple(dummy.shape)} → output {tuple(out.shape)}")
