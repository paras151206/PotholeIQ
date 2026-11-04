"""
src/utils/plotting.py
----------------------
Reusable plotting helpers.  Every function saves a PNG to results/plots/
AND returns the matplotlib Figure so callers can embed it in notebooks.

Design choice: We use matplotlib (not seaborn for the core plots) so the
code is easier to understand line-by-line in a viva.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses:   Sequence[float],
    train_accs:   Sequence[float],
    val_accs:     Sequence[float],
    title: str = "Training Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot loss and accuracy curves side-by-side.
    A gap between train/val curves is the visual signature of overfitting.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # -- Loss --
    ax1.plot(epochs, train_losses, label="Train loss", linewidth=2)
    ax1.plot(epochs, val_losses,   label="Val loss",   linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -- Accuracy --
    ax2.plot(epochs, train_accs, label="Train acc", linewidth=2)
    ax2.plot(epochs, val_accs,   label="Val acc",   linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved -> {save_path}")

    return fig


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a labelled confusion matrix heatmap."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved -> {save_path}")

    return fig


def plot_roc_curve(
    fpr: Sequence[float],
    tpr: Sequence[float],
    auc_score: float,
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a ROC curve (used for the autoencoder anomaly detector)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved -> {save_path}")

    return fig

