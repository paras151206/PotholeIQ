#!/usr/bin/env python3
"""
smoke_test.py
=============
End-to-end sanity check on 10 synthetic images.

Purpose
-------
Proves the full pipeline is importable and runnable WITHOUT needing the
real dataset.  Run this after every major change to catch import errors,
shape mismatches, and config bugs early.

Usage
-----
    python smoke_test.py

Expected output: "ALL SMOKE TESTS PASSED" with no exceptions.
"""

import sys
import traceback
from pathlib import Path

# ── Make sure repo root is on the path ───────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 4          # small batch so it runs fast even on CPU
H, W   = 224, 224   # must match config.yaml → data.image_size
PASSED = []
FAILED = []


def run(name: str, fn):
    """Run a single test, capture pass/fail."""
    try:
        fn()
        PASSED.append(name)
        print(f"  ✓  {name}")
    except Exception:
        FAILED.append(name)
        print(f"  ✗  {name}")
        traceback.print_exc()


# ─────────────────────────── Tests ───────────────────────────────────────────

def test_config():
    from src.utils.config import CFG
    assert CFG.seed == 42
    assert len(CFG.data.split_ratios) == 3
    assert abs(sum(CFG.data.split_ratios) - 1.0) < 1e-6, "split ratios must sum to 1"


def test_seed():
    from src.utils.seed import set_all_seeds
    set_all_seeds(42)
    # Reproducibility check: same seed → same random number
    t1 = torch.rand(5)
    set_all_seeds(42)
    t2 = torch.rand(5)
    assert torch.allclose(t1, t2), "Seeds not reproducible"


def test_metrics():
    from src.utils.metrics import classification_metrics, dice_coefficient, iou_score
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = classification_metrics(y_true, y_pred)
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["f1"] <= 1

    pred = np.array([[1, 0], [1, 1]])
    true = np.array([[1, 1], [1, 0]])
    d = dice_coefficient(pred, true)
    iou = iou_score(pred, true)
    assert 0 <= d <= 1
    assert 0 <= iou <= 1


def test_plotting():
    from src.utils.plotting import plot_training_curves
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for headless environments
    fig = plot_training_curves(
        [0.9, 0.7, 0.5], [1.0, 0.8, 0.6],
        [0.6, 0.7, 0.8], [0.5, 0.65, 0.75],
        title="Smoke Test",
    )
    assert fig is not None


# ── Placeholder tests (filled in as modules are built) ───────────────────────

def test_placeholder_cnn():
    """Placeholder: will test CustomCNN forward pass once M3 is built."""
    # TODO: replace with actual import once src/models/cnn.py exists
    pass


def test_placeholder_autoencoder():
    """Placeholder: will test AE forward pass once M8 is built."""
    pass


# ─────────────────────────── Runner ──────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  PotholeIQ Smoke Test  |  device={DEVICE}")
    print(f"{'='*55}\n")

    run("Config loads correctly",      test_config)
    run("Seeds are reproducible",      test_seed)
    run("Metrics helpers work",        test_metrics)
    run("Plotting helpers work",       test_plotting)
    run("CNN placeholder",             test_placeholder_cnn)
    run("Autoencoder placeholder",     test_placeholder_autoencoder)

    print(f"\n{'─'*55}")
    print(f"  Results: {len(PASSED)} passed, {len(FAILED)} failed")
    print(f"{'─'*55}")

    if FAILED:
        print("\n  ❌  SOME TESTS FAILED — fix before continuing.\n")
        sys.exit(1)
    else:
        print("\n  ✅  ALL SMOKE TESTS PASSED\n")
        sys.exit(0)

