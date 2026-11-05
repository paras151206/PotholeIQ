"""
src/utils/metrics.py
---------------------
Reusable metric helpers used across classifiers, segmentation, and
the autoencoder.  Every function returns plain Python scalars or dicts
so they can be JSON-serialised into results/metrics/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------- Classification ----------------------------------

def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
    class_names: list[str] | None = None,
    model_name: str | None = None,   # optional label stored in the metrics dict
) -> dict:
    """
    Compute accuracy, precision, recall, F1, confusion matrix, and
    optionally ROC-AUC (requires y_prob for the positive class).

    Parameters
    ----------
    y_true      : ground-truth integer labels
    y_pred      : predicted integer labels
    y_prob      : predicted probability for the positive class (binary only)
    class_names : human-readable class names for the report

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                    roc_auc (if y_prob provided)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "model_name": model_name,   # may be None; useful when comparing multiple runs
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred,
            target_names=class_names or [str(i) for i in np.unique(y_true)],
            zero_division=0,
        ),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None   # e.g. only one class in y_true

    return metrics


# --------------------------- Segmentation ------------------------------------

def dice_coefficient(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Both masks should be binary (0/1) arrays of the same shape.
    """
    pred = pred_mask.flatten().astype(float)
    true = true_mask.flatten().astype(float)
    intersection = (pred * true).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + true.sum() + smooth))


def iou_score(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    IoU (Jaccard) = |X ∩ Y| / |X ∪ Y|
    """
    pred = pred_mask.flatten().astype(float)
    true = true_mask.flatten().astype(float)
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


# --------------------------- I/O helpers -------------------------------------

def save_metrics(metrics: dict, path: str | Path) -> None:
    """Serialise metrics dict to JSON (skips non-serialisable values)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python natives for JSON
    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serialisable = {k: _convert(v) for k, v in metrics.items() if k != "report"}
    # Save report as plain text alongside the JSON
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)

    if "report" in metrics:
        report_path = path.with_suffix(".txt")
        with open(report_path, "w") as f:
            f.write(metrics["report"])

    print(f"[metrics] Saved -> {path}")

