"""
train_siamese.py — Siamese network training script (Module M7)

Usage:
    python src/models/train_siamese.py

What this trains:
    A Siamese network that learns to embed images into a 128-dim space
    where same-type images are close and different-type images are far.

    Positive pairs  (label=0, same): augmented views of the same pothole image.
    Negative pairs  (label=1, diff): a pothole image paired with a normal image.

    Loss: Contrastive loss
        L = (1-y)*d^2 + y*max(0, margin - d)^2
        where d = ||emb1 - emb2||_2

Outputs:
    results/checkpoints/siamese_best.pth
    results/metrics/siamese_test_metrics.json
    results/plots/siamese_training.png
    results/plots/siamese_tsne.png  (t-SNE of embeddings on test set)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_all_seeds, CFG, worker_init_fn, save_metrics
from src.data.dataset import SiameseDataset, PotholeDataset, get_transforms
from src.models.siamese import SiameseNet, ContrastiveLoss, build_siamese

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME  = "siamese"
SPLITS_DIR  = ROOT / CFG.paths.splits_dir
CKPT_DIR    = ROOT / CFG.paths.checkpoints_dir
PLOTS_DIR   = ROOT / CFG.paths.plots_dir
METRICS_DIR = ROOT / CFG.paths.metrics_dir
for d in [CKPT_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EPOCHS    = CFG.siamese.epochs
PATIENCE  = CFG.train.early_stopping_patience
MARGIN    = CFG.siamese.margin
BATCH     = CFG.train.batch_size


# ─────────────────────────── Train / eval helpers ─────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, margin):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for img1, img2, labels in loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img1.size(0)
        # Pair accuracy: same if dist < margin/2, diff if dist >= margin/2
        dist = torch.norm(emb1 - emb2, dim=1)
        pred_same  = (dist < margin / 2).long()
        correct   += (pred_same == (1 - labels)).sum().item()
        total     += img1.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device, margin):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for img1, img2, labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels.float())
            total_loss += loss.item() * img1.size(0)
            dist = torch.norm(emb1 - emb2, dim=1)
            pred_same  = (dist < margin / 2).long()
            correct   += (pred_same == (1 - labels)).sum().item()
            total     += img1.size(0)
    return total_loss / total, correct / total


# ─────────────────────────── t-SNE visualisation ─────────────────────────────

def plot_tsne(model, csv_path, device, save_path, max_samples=300):
    """
    Embed up to max_samples test images and visualise with t-SNE.
    Red = pothole, Blue = normal.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[siamese] sklearn not available — skipping t-SNE.")
        return

    tf  = get_transforms("val")
    ds  = PotholeDataset(csv_path, transform=tf)
    n   = min(max_samples, len(ds))
    idx = np.random.choice(len(ds), n, replace=False)

    embeddings, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for i in idx:
            img, lbl = ds[i]
            emb = model.forward_one(img.unsqueeze(0).to(device))
            embeddings.append(emb.cpu().numpy()[0])
            labels_all.append(lbl)

    embeddings = np.array(embeddings)   # (n, embedding_dim)
    labels_all = np.array(labels_all)

    # t-SNE: reduce to 2D
    # perplexity must be < n_samples
    perplexity = min(30, n - 1)
    tsne = TSNE(n_components=2, random_state=CFG.seed, perplexity=perplexity)
    reduced = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, color, name in [(0, "steelblue", "normal"), (1, "tomato", "pothole")]:
        mask = labels_all == lbl
        ax.scatter(reduced[mask, 0], reduced[mask, 1], c=color, label=name,
                   alpha=0.7, s=20, edgecolors="none")
    ax.set_title("Siamese Embeddings — t-SNE (test set)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[siamese] t-SNE saved → {save_path}")


# ─────────────────────────── Main ─────────────────────────────────────────────

def main():
    set_all_seeds(CFG.seed)
    print(f"\n{'='*60}")
    print(f"  PotholeIQ — M7 Siamese Network | device={DEVICE}")
    print(f"  NOTE: Positive pairs = augmented views of same image,")
    print(f"        NOT real same-pothole-over-time pairs (see README).")
    print(f"{'='*60}\n")

    for split in ["train", "val", "test"]:
        p = SPLITS_DIR / f"{split}.csv"
        if not p.exists():
            print(f"[siamese] ERROR: {p} not found. Run src/data/split.py first.")
            sys.exit(1)

    # ── Datasets ──────────────────────────────────────────────────────────────
    val_tf = get_transforms("val")
    train_ds = SiameseDataset(SPLITS_DIR / "train.csv", transform=val_tf)
    val_ds   = SiameseDataset(SPLITS_DIR / "val.csv",   transform=val_tf)

    loader_kw = dict(batch_size=BATCH, num_workers=CFG.data.num_workers,
                     pin_memory=CFG.data.pin_memory, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_siamese().to(DEVICE)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.siamese.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    pat_count     = 0
    ckpt_path     = CKPT_DIR / "siamese_best.pth"
    history       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, MARGIN)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE, MARGIN)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"Ep {epoch+1:3d}/{EPOCHS} | "
              f"tr_loss={tr_loss:.4f} tr_pair_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_pair_acc={val_acc:.4f}")

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
    epochs_ran = len(history["train_loss"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, epochs_ran + 1)
    ax1.plot(ep, history["train_loss"], label="Train", lw=2)
    ax1.plot(ep, history["val_loss"],   label="Val",   lw=2, ls="--")
    ax1.set(title="Siamese — Contrastive Loss", xlabel="Epoch", ylabel="Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, history["train_acc"], label="Train pair acc", lw=2)
    ax2.plot(ep, history["val_acc"],   label="Val pair acc",   lw=2, ls="--")
    ax2.set(title="Siamese — Pair Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "siamese_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    test_ds  = SiameseDataset(SPLITS_DIR / "test.csv", transform=val_tf)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, DEVICE, MARGIN)

    # Compute same/different distances for analysis
    model.eval()
    same_dists, diff_dists = [], []
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            emb1, emb2 = model(img1, img2)
            dists = torch.norm(emb1 - emb2, dim=1).cpu().numpy()
            for d, l in zip(dists, labels.numpy()):
                if l == 0:
                    same_dists.append(float(d))
                else:
                    diff_dists.append(float(d))

    metrics = {
        "test_contrastive_loss": float(test_loss),
        "test_pair_accuracy":    float(test_acc),
        "avg_dist_same_pair":    float(np.mean(same_dists)) if same_dists else None,
        "avg_dist_diff_pair":    float(np.mean(diff_dists)) if diff_dists else None,
        "margin":                MARGIN,
        "note": "Positive pairs are augmented views of same image, NOT real same-pothole-over-time pairs",
    }
    save_metrics(metrics, METRICS_DIR / "siamese_test_metrics.json")

    print(f"\n{'─'*60}")
    print(f"  [siamese] TEST RESULTS")
    print(f"{'─'*60}")
    print(f"  Contrastive loss : {test_loss:.4f}")
    print(f"  Pair accuracy    : {test_acc:.4f}")
    if same_dists:
        print(f"  Avg dist (same)  : {np.mean(same_dists):.4f}")
    if diff_dists:
        print(f"  Avg dist (diff)  : {np.mean(diff_dists):.4f}")
    print(f"{'─'*60}\n")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    plot_tsne(model, SPLITS_DIR / "test.csv", DEVICE,
              PLOTS_DIR / "siamese_tsne.png", max_samples=300)


if __name__ == "__main__":
    main()
