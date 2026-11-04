"""
src/models/train_unet.py
========================
Training script for the SmallUNet segmentation model.

Loss = dice_weight * DiceLoss + (1 - dice_weight) * BCELoss
     = 0.5 * (1 - dice_coeff) + 0.5 * BCE

The combined loss addresses the class-imbalance problem in segmentation
(potholes are small relative to the full road area): Dice is recall-aware
while BCE handles per-pixel binary cross-entropy.

Run:
    python src/models/train_unet.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import warnings

import matplotlib
matplotlib.use("Agg")  # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import (
    CFG,
    set_all_seeds,
    worker_init_fn,
    dice_coefficient,
    iou_score,
    save_metrics,
    plot_training_curves,
)
from src.data.dataset import PotholeSegDataset, get_transforms
from src.models.unet import build_unet, compute_severity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "unet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLITS_DIR   = ROOT / CFG.paths.splits_dir
PLOTS_DIR    = ROOT / CFG.paths.plots_dir
METRICS_DIR  = ROOT / CFG.paths.metrics_dir
CKPT_DIR     = ROOT / CFG.paths.checkpoints_dir

for d in (PLOTS_DIR, METRICS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

BEST_CKPT = CKPT_DIR / "unet_best.pth"


# ---------------------------------------------------------------------------
# Fallback: build a seg CSV from classification train.csv when needed
# ---------------------------------------------------------------------------
def _ensure_seg_csv(split: str) -> Path:
    """Return path to seg_{split}.csv, creating it from {split}.csv if absent.

    When proper segmentation annotations are unavailable the script uses the
    classification split as a proxy.  Pothole images get a placeholder bbox_path
    pointing to the labels/ folder so the dataset loader can handle the
    missing annotation gracefully (it fills in a zero mask).
    """
    seg_path = SPLITS_DIR / f"seg_{split}.csv"
    if seg_path.exists():
        return seg_path

    cls_path = SPLITS_DIR / f"{split}.csv"
    if not cls_path.exists():
        print(f"[ERROR] Neither {seg_path} nor {cls_path} found.")
        print("Run src/data/split.py first.")
        sys.exit(1)

    warnings.warn(
        f"{seg_path.name} not found – generating proxy from {cls_path.name}. "
        "All masks will be zero; real segmentation training requires proper "
        "annotation files.",
        UserWarning,
        stacklevel=2,
    )

    import pandas as pd
    df = pd.read_csv(cls_path)
    # bbox_path column: point to a labels directory; PotholeSegDataset should
    # return a zero mask when the actual annotation file is absent.
    df["bbox_path"] = str(ROOT / "data" / "labels")
    df.to_csv(seg_path, index=False)
    print(f"[INFO] Created proxy {seg_path}")
    return seg_path


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable Dice loss.  1 – Dice_coefficient.

    Smooth=1 is added to numerator and denominator to avoid division by zero
    when both pred and target are all-zero (background-only patches).
    """
    return 1.0 - dice_coefficient(pred, target)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dice_w: float = CFG.unet.dice_weight,
) -> torch.Tensor:
    """Weighted sum of Dice loss and binary cross-entropy.

    dice_w=0.5 means equal contribution from both terms, which works well for
    moderately imbalanced segmentation tasks.
    """
    bce  = nn.BCELoss()(pred, target)
    dl   = dice_loss(pred, target)
    return dice_w * dl + (1.0 - dice_w) * bce


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimiser, training: bool):
    """One forward (+ optional backward) pass over *loader*.

    Returns average loss, average Dice, average IoU for the epoch.
    """
    model.train(training)
    total_loss = total_dice = total_iou = 0.0
    n = 0

    with torch.set_grad_enabled(training):
        for imgs, masks in loader:
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)          # (B,1,H,W) float

            preds = model(imgs)               # (B,1,H,W) ∈ [0,1]
            loss  = combined_loss(preds, masks)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            # Threshold at 0.5 for discrete metric computation
            binary = (preds > 0.5).float()
            total_loss += loss.item() * imgs.size(0)
            total_dice += dice_coefficient(binary, masks).item() * imgs.size(0)
            total_iou  += iou_score(binary, masks).item() * imgs.size(0)
            n += imgs.size(0)

    return total_loss / n, total_dice / n, total_iou / n


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def save_prediction_grid(model, test_loader, num_samples: int = 5):
    """Save a side-by-side grid: original | predicted mask | severity label."""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Predicted Mask")
    axes[0, 2].set_title("Severity")

    mean = np.array(CFG.augmentation.normalize_mean)
    std  = np.array(CFG.augmentation.normalize_std)

    count = 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs)   # (B,1,H,W)

            for b in range(imgs.size(0)):
                if count >= num_samples:
                    break
                # De-normalise for display
                img_np = imgs[b].cpu().permute(1, 2, 0).numpy()
                img_np = np.clip(img_np * std + mean, 0, 1)

                pred_mask = preds[b].cpu().squeeze().numpy()
                sev = compute_severity(preds[b].cpu())

                axes[count, 0].imshow(img_np)
                axes[count, 0].axis("off")
                axes[count, 1].imshow(pred_mask, cmap="hot", vmin=0, vmax=1)
                axes[count, 1].axis("off")
                axes[count, 2].text(
                    0.5, 0.5, sev,
                    ha="center", va="center", fontsize=16,
                    color={"Low": "green", "Medium": "orange", "High": "red"}.get(sev, "black"),
                    transform=axes[count, 2].transAxes,
                )
                axes[count, 2].axis("off")
                count += 1

            if count >= num_samples:
                break

    plt.tight_layout()
    out_path = PLOTS_DIR / "unet_predictions.png"
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"[INFO] Saved prediction grid -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_all_seeds(CFG.seed)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] U-Net training – epochs={CFG.unet.epochs}, lr={CFG.unet.lr}")

    # ---- Data --------------------------------------------------------------
    train_csv = _ensure_seg_csv("train")
    val_csv   = _ensure_seg_csv("val")
    test_csv  = _ensure_seg_csv("test")

    img_tf = get_transforms("train")
    val_tf = get_transforms("val")

    train_ds = PotholeSegDataset(train_csv, img_transform=img_tf)
    val_ds   = PotholeSegDataset(val_csv,   img_transform=val_tf)
    test_ds  = PotholeSegDataset(test_csv,  img_transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.train.batch_size,
        shuffle=True,
        num_workers=CFG.data.num_workers,
        pin_memory=CFG.data.pin_memory,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.train.batch_size,
        shuffle=False,
        num_workers=CFG.data.num_workers,
        pin_memory=CFG.data.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.train.batch_size,
        shuffle=False,
        num_workers=CFG.data.num_workers,
        pin_memory=CFG.data.pin_memory,
    )

    print(f"[INFO] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    # ---- Model, optimiser --------------------------------------------------
    model     = build_unet().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=CFG.unet.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3, verbose=True
    )

    # ---- Training loop with early stopping ---------------------------------
    history = {k: [] for k in ["train_loss", "val_loss", "train_dice", "val_dice"]}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = CFG.train.early_stopping_patience

    for epoch in range(1, CFG.unet.epochs + 1):
        tr_loss, tr_dice, tr_iou = run_epoch(model, train_loader, optimiser, training=True)
        va_loss, va_dice, va_iou = run_epoch(model, val_loader,   optimiser, training=False)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)

        print(
            f"Epoch {epoch:02d}/{CFG.unet.epochs} | "
            f"tr_loss={tr_loss:.4f} dice={tr_dice:.4f} | "
            f"va_loss={va_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_CKPT)
            print(f"  -> Saved best checkpoint (val_loss={va_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # ---- Test evaluation ---------------------------------------------------
    model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))
    te_loss, te_dice, te_iou = run_epoch(model, test_loader, None, training=False)
    print(f"\n[TEST] loss={te_loss:.4f} | Dice={te_dice:.4f} | IoU={te_iou:.4f}")

    # ---- Severity distribution on test set --------------------------------
    model.eval()
    sev_counts = {"Low": 0, "Medium": 0, "High": 0}
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs)
            for b in range(preds.size(0)):
                sev = compute_severity(preds[b].cpu())
                sev_counts[sev] += 1

    total_test = sum(sev_counts.values())
    print("\n[Severity Distribution – Test Set]")
    for sev, cnt in sev_counts.items():
        pct = 100.0 * cnt / max(total_test, 1)
        print(f"  {sev:6s}: {cnt} ({pct:.1f}%)")

    # ---- Save metrics ------------------------------------------------------
    metrics = {
        "test_loss": round(te_loss, 6),
        "test_dice": round(te_dice, 6),
        "test_iou":  round(te_iou, 6),
        "severity_distribution": sev_counts,
    }
    save_metrics(metrics, METRICS_DIR / "unet_metrics.json")
    print(f"[INFO] Metrics saved -> {METRICS_DIR / 'unet_metrics.json'}")

    # ---- Training curves ---------------------------------------------------
    plot_training_curves(
        history,
        title="U-Net Training Curves",
        save_path=str(PLOTS_DIR / "unet_training.png"),
    )
    print(f"[INFO] Training plot saved -> {PLOTS_DIR / 'unet_training.png'}")

    # ---- Prediction visualisation -----------------------------------------
    save_prediction_grid(model, test_loader, num_samples=5)

    print("\n[DONE] U-Net training complete.")


if __name__ == "__main__":
    main()
