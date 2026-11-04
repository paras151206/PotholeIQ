"""
train_transfer.py — Two-phase MobileNetV2 fine-tuning script (Module M4)

Usage:
    python src/models/train_transfer.py

Phase 1 (frozen backbone):
    - Only the new classifier head is trained.
    - Higher LR (0.001) is safe because the backbone is frozen.
    - Runs for CFG.transfer.phase1_epochs epochs.

Phase 2 (partial unfreeze):
    - Last CFG.transfer.unfreeze_blocks InvertedResidual blocks are unfrozen.
    - Much lower LR (0.00005) to avoid catastrophic forgetting.
    - Runs for CFG.transfer.phase2_epochs epochs, starting from phase1 weights.

The best checkpoint from each phase is saved separately.
Test evaluation uses the best phase2 checkpoint (once only).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")

from src.utils import set_all_seeds, CFG, worker_init_fn
from src.utils import classification_metrics, save_metrics
from src.utils import plot_training_curves, plot_confusion_matrix
from src.data.dataset import PotholeDataset, get_transforms
from src.models.transfer import TransferModel, build_transfer_model

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "transfer"
SPLITS_DIR = ROOT / CFG.paths.splits_dir
CKPT_DIR   = ROOT / CFG.paths.checkpoints_dir
PLOTS_DIR  = ROOT / CFG.paths.plots_dir
METRICS_DIR = ROOT / CFG.paths.metrics_dir
for d in [CKPT_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PATIENCE = CFG.train.early_stopping_patience


# ─────────────────────────── Training helpers ────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels, all_probs


def train_phase(model, train_loader, val_loader, optimizer, scheduler, criterion,
                device, epochs, patience, checkpoint_path, phase_name):
    """Run one training phase with early stopping. Returns history dict."""
    best_val_loss = float("inf")
    pat_count     = 0
    history       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if scheduler:
            scheduler.step()

        print(f"  [{phase_name}] Ep {epoch+1:3d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pat_count     = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"            → Best checkpoint saved (val_loss={val_loss:.4f})")
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"  [{phase_name}] Early stopping at epoch {epoch+1}")
                break

    return history


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    set_all_seeds(CFG.seed)
    print(f"\n{'='*60}")
    print(f"  PotholeIQ — M4 Transfer Learning | device={DEVICE}")
    print(f"{'='*60}\n")

    # ── Verify splits exist ───────────────────────────────────────────────────
    for split in ["train", "val", "test"]:
        p = SPLITS_DIR / f"{split}.csv"
        if not p.exists():
            print(f"[transfer] ERROR: {p} not found. Run src/data/split.py first.")
            sys.exit(1)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_tf = get_transforms("train")
    val_tf   = get_transforms("val")

    train_ds = PotholeDataset(SPLITS_DIR / "train.csv", transform=train_tf)
    val_ds   = PotholeDataset(SPLITS_DIR / "val.csv",   transform=val_tf)
    test_ds  = PotholeDataset(SPLITS_DIR / "test.csv",  transform=val_tf)

    # Compute class weights to handle imbalance
    labels_arr = np.array(train_ds.labels)
    weights    = compute_class_weight("balanced", classes=np.unique(labels_arr), y=labels_arr)
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    print(f"[transfer] Class weights: normal={weights[0]:.3f}, pothole={weights[1]:.3f}")

    loader_kw = dict(batch_size=CFG.train.batch_size, num_workers=CFG.data.num_workers,
                     pin_memory=CFG.data.pin_memory, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_transfer_model(pretrained=True).to(DEVICE)
    print(f"[transfer] MobileNetV2 loaded (pretrained=True)")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Frozen backbone, train head only
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n── Phase 1: Frozen backbone ({CFG.transfer.phase1_epochs} epochs) ──")
    model.freeze_backbone()

    # Only pass head parameters to the optimizer
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.Adam(head_params, lr=CFG.transfer.phase1_lr, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=CFG.transfer.phase1_epochs)
    ckpt1 = CKPT_DIR / "transfer_phase1_best.pth"

    h1 = train_phase(model, train_loader, val_loader, opt1, sch1, criterion,
                     DEVICE, CFG.transfer.phase1_epochs, PATIENCE, ckpt1, "Phase1")

    # Save phase1 metrics on val set
    model.load_state_dict(torch.load(ckpt1, map_location=DEVICE))
    _, _, p1_preds, p1_labels, p1_probs = eval_epoch(model, val_loader, criterion, DEVICE)
    p1_metrics = classification_metrics(p1_labels, p1_preds, p1_probs, ["normal", "pothole"])
    save_metrics(p1_metrics, METRICS_DIR / "transfer_phase1_metrics.json")
    print(f"[transfer] Phase 1 val F1={p1_metrics['f1']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Unfreeze last blocks, fine-tune with low LR
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n── Phase 2: Unfreezing last {CFG.transfer.unfreeze_blocks} blocks "
          f"({CFG.transfer.phase2_epochs} epochs) ──")
    model.unfreeze_last_blocks(CFG.transfer.unfreeze_blocks)

    # Optimizer now covers all unfrozen parameters
    opt2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.transfer.phase2_lr,
        weight_decay=1e-4,
    )
    sch2  = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CFG.transfer.phase2_epochs)
    ckpt2 = CKPT_DIR / "transfer_best.pth"

    h2 = train_phase(model, train_loader, val_loader, opt2, sch2, criterion,
                     DEVICE, CFG.transfer.phase2_epochs, PATIENCE, ckpt2, "Phase2")

    # ── Save combined training curves ─────────────────────────────────────────
    # Concatenate phase1 + phase2 histories for the plot
    combined_tr_loss  = h1["train_loss"] + h2["train_loss"]
    combined_val_loss = h1["val_loss"]   + h2["val_loss"]
    combined_tr_acc   = h1["train_acc"]  + h2["train_acc"]
    combined_val_acc  = h1["val_acc"]    + h2["val_acc"]

    plot_training_curves(
        combined_tr_loss, combined_val_loss,
        combined_tr_acc,  combined_val_acc,
        title="Transfer Learning (Phase1 | Phase2)",
        save_path=PLOTS_DIR / "transfer_training.png",
    )

    # ── Test evaluation (ONCE, using best phase2 checkpoint) ─────────────────
    model.load_state_dict(torch.load(ckpt2, map_location=DEVICE))
    _, _, test_preds, test_labels, test_probs = eval_epoch(model, test_loader, criterion, DEVICE)

    test_metrics = classification_metrics(test_labels, test_preds, test_probs, ["normal", "pothole"])
    save_metrics(test_metrics, METRICS_DIR / "transfer_test_metrics.json")

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=["normal", "pothole"],
        title="Transfer Learning — Confusion Matrix",
        save_path=PLOTS_DIR / "transfer_confusion_matrix.png",
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [transfer] TEST RESULTS")
    print(f"{'─'*60}")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    if test_metrics.get("roc_auc"):
        print(f"  ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
