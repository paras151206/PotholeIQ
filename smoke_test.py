#!/usr/bin/env python3
"""
smoke_test.py
=============
End-to-end sanity check — works WITHOUT the real dataset.
Runs on 10 synthetic images to prove all modules import correctly
and all model forward passes produce the right shapes.

Usage:
    python smoke_test.py

Expected: "ALL SMOKE TESTS PASSED" with no exceptions.
"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 2
H, W   = 128, 128
PASSED, FAILED = [], []


def run(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  [PASS]  {name}")
    except Exception:
        FAILED.append(name)
        print(f"  [FAIL]  {name}")
        traceback.print_exc()


# ─────────────────────────── Tests ───────────────────────────────────────────

def test_config():
    from src.utils.config import CFG
    assert CFG.seed == 42
    assert CFG.data.image_size == [128, 128]
    assert abs(sum(CFG.data.split_ratios) - 1.0) < 1e-6


def test_seed():
    from src.utils.seed import set_all_seeds
    set_all_seeds(42)
    t1 = torch.rand(5)
    set_all_seeds(42)
    t2 = torch.rand(5)
    assert torch.allclose(t1, t2)


def test_metrics():
    from src.utils.metrics import classification_metrics, dice_coefficient, iou_score
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert 0 <= m["accuracy"] <= 1
    pred = np.array([[1, 0], [1, 1]])
    true = np.array([[1, 1], [1, 0]])
    assert 0 <= dice_coefficient(pred, true) <= 1
    assert 0 <= iou_score(pred, true) <= 1


def test_plotting():
    import matplotlib
    matplotlib.use("Agg")
    from src.utils.plotting import plot_training_curves
    fig = plot_training_curves([0.9, 0.7], [1.0, 0.8], [0.6, 0.7], [0.5, 0.65])
    assert fig is not None


def test_baseline():
    from src.models.baseline import build_baseline
    model = build_baseline().to(DEVICE)
    x = torch.randn(BATCH, 3, H, W).to(DEVICE)
    y = model(x)
    assert y.shape == (BATCH, 2), f"Expected (B,2), got {y.shape}"


def test_cnn():
    from src.models.cnn import build_cnn, build_cnn_no_bn, build_cnn_no_dropout
    for name, builder in [("full", build_cnn), ("no_bn", build_cnn_no_bn), ("no_drop", build_cnn_no_dropout)]:
        model = builder().to(DEVICE)
        x = torch.randn(BATCH, 3, H, W).to(DEVICE)
        y = model(x)
        assert y.shape == (BATCH, 2), f"CNN [{name}] shape mismatch: {y.shape}"


def test_transfer():
    from src.models.transfer import build_transfer_model
    model = build_transfer_model(pretrained=False).to(DEVICE)
    x = torch.randn(BATCH, 3, H, W).to(DEVICE)
    y = model(x)
    assert y.shape == (BATCH, 2)
    # Test freeze/unfreeze
    model.freeze_backbone()
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    model.unfreeze_last_blocks(2)
    unfrozen_after = sum(1 for p in model.parameters() if p.requires_grad)
    assert unfrozen_after > 0


def test_unet():
    from src.models.unet import build_unet, compute_severity
    model = build_unet().to(DEVICE)
    x = torch.randn(BATCH, 3, H, W).to(DEVICE)
    y = model(x)
    assert y.shape == (BATCH, 1, H, W), f"UNet shape mismatch: {y.shape}"
    assert y.min() >= 0 and y.max() <= 1, "UNet output not in [0,1]"
    # Test severity
    from src.utils.config import CFG
    sev = compute_severity(y[0], CFG.unet.severity_thresholds)
    assert sev in ["Low", "Medium", "High"]


def test_siamese():
    from src.models.siamese import build_siamese, ContrastiveLoss
    model = build_siamese().to(DEVICE)
    x1 = torch.randn(BATCH, 3, H, W).to(DEVICE)
    x2 = torch.randn(BATCH, 3, H, W).to(DEVICE)
    e1, e2 = model(x1, x2)
    assert e1.shape == (BATCH, 128)
    # L2 norm should be ~1
    norms = torch.norm(e1, dim=1)
    assert torch.allclose(norms, torch.ones(BATCH, device=DEVICE), atol=1e-5)
    # Contrastive loss should be non-negative
    loss_fn = ContrastiveLoss(margin=1.0)
    labels  = torch.zeros(BATCH).to(DEVICE)   # same pair
    loss    = loss_fn(e1, e2, labels)
    assert loss.item() >= 0


def test_autoencoder():
    from src.models.autoencoder import build_autoencoder, compute_anomaly_score
    model = build_autoencoder().to(DEVICE)
    x = torch.randn(BATCH, 3, H, W).to(DEVICE)
    recon = model(x)
    assert recon.shape == (BATCH, 3, H, W), f"AE shape mismatch: {recon.shape}"
    score = compute_anomaly_score(model, x[0], torch.device(DEVICE))
    assert score >= 0


def test_gradcam_import():
    from src.explainability.gradcam import GradCAM
    # Just test import — full run needs a checkpoint
    assert GradCAM is not None


def test_yolo_detector():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    dummy = np.zeros((416, 416, 3), dtype=np.uint8)
    res = model(dummy, verbose=False)
    assert len(res) == 1
    assert hasattr(res[0], "boxes")


# ─────────────────────────── Runner ──────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  PotholeIQ Smoke Test  |  device={DEVICE}")
    print(f"{'='*55}\n")

    run("Config loads",              test_config)
    run("Seeds reproducible",        test_seed)
    run("Metrics helpers",           test_metrics)
    run("Plotting helpers",          test_plotting)
    run("Baseline MLP",              test_baseline)
    run("Custom CNN (3 variants)",   test_cnn)
    run("Transfer MobileNetV2",      test_transfer)
    run("U-Net segmentation",        test_unet)
    run("Siamese + ContrastiveLoss", test_siamese)
    run("Convolutional Autoencoder", test_autoencoder)
    run("Grad-CAM import",           test_gradcam_import)
    run("YOLOv8n Object Detector",   test_yolo_detector)

    print(f"\n{'=' * 55}")
    print(f"  Results: {len(PASSED)} passed, {len(FAILED)} failed")
    print(f"{'=' * 55}")

    if FAILED:
        print("\n  [FAILED] SOME TESTS FAILED\n")
        sys.exit(1)
    else:
        print("\n  [OK] ALL SMOKE TESTS PASSED\n")
        sys.exit(0)
