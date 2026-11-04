"""
train_autoencoder.py — Convolutional Autoencoder training script (Module M8)

Usage:
    python src/models/train_autoencoder.py

Training philosophy:
    The autoencoder is an UNSUPERVISED model — it sees NO labels during training.
    It learns what "normal road" looks like by minimising reconstruction error
    on normal images only.

    At test time:
        - Normal images → low reconstruction error (model knows what they look like)
        - Pothole images → high reconstruction error (never seen during training)
    This reconstruction error is the "anomaly score".

Outputs:
    results/checkpoints/autoencoder_best.pth
    results/metrics/autoencoder_test_metrics.json
    results/plots/autoencoder_training.png
    results/plots/autoencoder_roc.png
    results/plots/autoencoder_score_distribution.png
    results/plots/autoencoder_reconstructions.png
    config.yaml → autoencoder.anomaly_threshold updated with best threshold
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import roc_auc_score, roc_curve, f1_score

from src.utils import set_all_seeds, CFG, worker_init_fn, save_metrics, plot_roc_curve
from src.data.dataset import NormalRoadDataset, PotholeDataset, get_transforms
from src.models.autoencoder import ConvAutoencoder, build_autoencoder, compute_anomaly_score

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME  = "autoencoder"
SPLITS_DIR  = ROOT / CFG.paths.splits_dir
CKPT_DIR    = ROOT / CFG.paths.checkpoints_dir
PLOTS_DIR   = ROOT / CFG.paths.plots_dir
METRICS_DIR = ROOT / CFG.paths.metrics_dir
for d in [CKPT_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EPOCHS   = CFG.autoencoder.epochs
PATIENCE = CFG.train.early_stopping_patience
BATCH    = CFG.train.batch_size


# ─────────────────────────── Training helpers ─────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    """AE train step: input == target, minimise reconstruction loss."""
    model.train()
    total_loss, total = 0.0, 0
    for imgs in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon = model(imgs)
        loss  = criterion(recon, imgs)   # reconstruct the input itself
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
    return total_loss / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for imgs in loader:
            imgs  = imgs.to(device)
            recon = model(imgs)
            loss  = criterion(recon, imgs)
            total_loss += loss.item() * imgs.size(0)
            total      += imgs.size(0)
    return total_loss / total


def compute_scores_for_dataset(model, csv_path, device, is_normal_only=False):
    """
    Compute anomaly score (MSE) for every image in a split CSV.
    Returns (scores, labels) arrays.
    """
    tf  = get_transforms("val")
    ds  = PotholeDataset(csv_path, transform=tf)
    scores, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for img, lbl in ds:
            score = compute_anomaly_score(model, img, device)
            scores.append(score)
            labels_all.append(int(lbl))
    return np.array(scores), np.array(labels_all)


# ─────────────────────────── Threshold selection ──────────────────────────────

def find_best_threshold(scores_val, labels_val, n_thresholds=50):
    """
    Try n_thresholds evenly-spaced values between min and max val scores.
    Pick the threshold that maximises F1 score on the validation set.
    Label convention: score > threshold → anomaly (pothole=1).
    """
    thresholds = np.linspace(scores_val.min(), scores_val.max(), n_thresholds)
    best_f1, best_thresh = -1.0, thresholds[len(thresholds) // 2]

    for t in thresholds:
        preds = (scores_val > t).astype(int)
        f1    = f1_score(labels_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t

    return float(best_thresh), float(best_f1)


# ─────────────────────────── Visualisations ───────────────────────────────────

def plot_score_distribution(scores_normal, scores_pothole, threshold, save_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(min(scores_normal.min(), scores_pothole.min()),
                       max(scores_normal.max(), scores_pothole.max()), 50)
    ax.hist(scores_normal,  bins=bins, alpha=0.6, color="steelblue", label="Normal",  edgecolor="black")
    ax.hist(scores_pothole, bins=bins, alpha=0.6, color="tomato",    label="Pothole", edgecolor="black")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.5f}")
    ax.set(title="Anomaly Score Distribution (test set)",
           xlabel="MSE Reconstruction Error", ylabel="Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[autoencoder] Saved → {save_path}")


def plot_reconstructions(model, csv_path, device, save_path, n_normal=3, n_pothole=2):
    """Show original vs reconstructed images for normal + pothole samples."""
    tf = get_transforms("val")
    ds = PotholeDataset(csv_path, transform=tf)

    normal_idxs  = [i for i in range(len(ds)) if ds.df.iloc[i]["label"] == 0][:n_normal]
    pothole_idxs = [i for i in range(len(ds)) if ds.df.iloc[i]["label"] == 1][:n_pothole]
    sample_idxs  = normal_idxs + pothole_idxs
    n_total      = len(sample_idxs)

    # Denormalise tensor back to uint8 image
    mean = torch.tensor(CFG.augmentation.normalize_mean).view(3, 1, 1)
    std  = torch.tensor(CFG.augmentation.normalize_std).view(3, 1, 1)
    def denorm(t):
        img = (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        return (img * 255).astype(np.uint8)

    fig, axes = plt.subplots(n_total, 2, figsize=(6, 2.5 * n_total))
    if n_total == 1:
        axes = [axes]
    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(sample_idxs):
            img, lbl = ds[idx]
            recon = model(img.unsqueeze(0).to(device)).squeeze(0).cpu()
            orig_np  = denorm(img)
            recon_np = denorm(recon)
            score    = compute_anomaly_score(model, img, device)
            cls_name = "normal" if lbl == 0 else "pothole"
            axes[row][0].imshow(orig_np);  axes[row][0].set_title(f"Original ({cls_name})", fontsize=8); axes[row][0].axis("off")
            axes[row][1].imshow(recon_np); axes[row][1].set_title(f"Reconstructed (MSE={score:.5f})", fontsize=8); axes[row][1].axis("off")

    fig.suptitle("Autoencoder Reconstructions", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[autoencoder] Saved → {save_path}")


# ─────────────────────────── Main ─────────────────────────────────────────────

def main():
    set_all_seeds(CFG.seed)
    print(f"\n{'='*60}")
    print(f"  PotholeIQ — M8 Convolutional Autoencoder | device={DEVICE}")
    print(f"  UNSUPERVISED: trained on normal images only.")
    print(f"{'='*60}\n")

    for split in ["train", "val", "test"]:
        p = SPLITS_DIR / f"{split}.csv"
        if not p.exists():
            print(f"[autoencoder] ERROR: {p} not found. Run src/data/split.py first.")
            sys.exit(1)

    # ── Datasets — normal images only for training ────────────────────────────
    tf = get_transforms("val")   # no augmentation (AE trains on clean images)
    train_ds = NormalRoadDataset(SPLITS_DIR / "train.csv", transform=tf)
    val_ds   = NormalRoadDataset(SPLITS_DIR / "val.csv",   transform=tf)

    loader_kw = dict(batch_size=BATCH, num_workers=CFG.data.num_workers,
                     pin_memory=CFG.data.pin_memory, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_autoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.autoencoder.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    pat_count     = 0
    ckpt_path     = CKPT_DIR / "autoencoder_best.pth"
    tr_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        tr_loss  = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        tr_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"Ep {epoch+1:3d}/{EPOCHS} | tr_loss={tr_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pat_count     = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"           → Best checkpoint saved")
        else:
            pat_count += 1
            if pat_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ── Training curves ───────────────────────────────────────────────────────
    ep = range(1, len(tr_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, tr_losses,  label="Train reconstruction loss", lw=2)
    ax.plot(ep, val_losses, label="Val reconstruction loss",   lw=2, ls="--")
    ax.set(title="Autoencoder — MSE Reconstruction Loss", xlabel="Epoch", ylabel="MSE")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "autoencoder_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Threshold selection on validation set ─────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print("\n[autoencoder] Computing anomaly scores on val set for threshold tuning...")
    val_scores, val_labels = compute_scores_for_dataset(
        model, SPLITS_DIR / "val.csv", DEVICE
    )
    threshold, val_f1 = find_best_threshold(val_scores, val_labels)
    print(f"[autoencoder] Best threshold (val F1={val_f1:.4f}): {threshold:.6f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("[autoencoder] Computing anomaly scores on test set...")
    test_scores, test_labels = compute_scores_for_dataset(
        model, SPLITS_DIR / "test.csv", DEVICE
    )

    test_preds = (test_scores > threshold).astype(int)

    # ROC-AUC
    try:
        auc = roc_auc_score(test_labels, test_scores)
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        plot_roc_curve(fpr, tpr, auc, title="Autoencoder ROC",
                       save_path=PLOTS_DIR / "autoencoder_roc.png")
    except ValueError as e:
        print(f"[autoencoder] ROC-AUC error: {e}")
        auc = None

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as skf1
    test_metrics = {
        "roc_auc":           float(auc) if auc is not None else None,
        "threshold":         float(threshold),
        "threshold_note":    "Selected to maximise F1 on val set (50 candidates)",
        "accuracy":          float(accuracy_score(test_labels, test_preds)),
        "precision":         float(precision_score(test_labels, test_preds, zero_division=0)),
        "recall":            float(recall_score(test_labels, test_preds, zero_division=0)),
        "f1":                float(skf1(test_labels, test_preds, zero_division=0)),
        "n_normal_test":     int((test_labels == 0).sum()),
        "n_pothole_test":    int((test_labels == 1).sum()),
    }
    save_metrics(test_metrics, METRICS_DIR / "autoencoder_test_metrics.json")

    # ── Score distribution plot ───────────────────────────────────────────────
    normal_scores  = test_scores[test_labels == 0]
    pothole_scores = test_scores[test_labels == 1]
    plot_score_distribution(normal_scores, pothole_scores, threshold,
                            PLOTS_DIR / "autoencoder_score_distribution.png")

    # ── Reconstruction plot ───────────────────────────────────────────────────
    plot_reconstructions(model, SPLITS_DIR / "test.csv", DEVICE,
                         PLOTS_DIR / "autoencoder_reconstructions.png")

    # ── Save threshold to config.yaml ─────────────────────────────────────────
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw["autoencoder"]["anomaly_threshold"] = float(threshold)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_raw, f, default_flow_style=False, sort_keys=False)
    print(f"[autoencoder] Updated config.yaml: anomaly_threshold={threshold:.6f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [autoencoder] TEST RESULTS")
    print(f"{'─'*60}")
    print(f"  ROC-AUC   : {auc:.4f}" if auc else "  ROC-AUC   : N/A")
    print(f"  Threshold : {threshold:.6f}")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
