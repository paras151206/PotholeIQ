"""
src/utils/bootstrap_ci.py
-------------------------
Computes 95% bootstrap confidence intervals for classification metrics (Accuracy & F1).
Sample size is N (test set size, 103 images).
Resampled with replacement 1000 times.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_bootstrap_ci(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    accs = []
    f1s = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_ci_95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "f1_mean": float(np.mean(f1s)),
        "f1_ci_95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }
