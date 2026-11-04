"""
train_cnn.py — Training script for CustomCNN + ablation studies (Module M3)

Usage:
    python src/models/train_cnn.py

What this script does:
1.  Main training run  ("cnn"):
      - Adam + CosineAnnealingLR
      - CrossEntropyLoss with label smoothing + class weights
      - Full augmentation on train split
      - Early stopping; saves best checkpoint

2.  Ablation study — No Augmentation  ("cnn_no_aug"):
      - Same model/optim/sched, but train split uses val-style transforms
      - Quantifies how much data augmentation helps

3.  Ablation study — No BatchNorm     ("cnn_no_bn"):
      - build_cnn_no_bn() model
      - Everything else identical to main run

4.  Ablation study — No Dropout       ("cnn_no_dropout"):
      - build_cnn_no_dropout() model
      - Everything else identical to main run

5.  Overfit sanity check:
      - 50-image subset, 20 epochs, no early stopping
      - Verifies the model can memorise small data (expect ~100% train acc)

Each run saves its own checkpoint, metrics JSON, training curve, and
confusion matrix so they can be compared in the final report.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset

from src.data.dataset import PotholeDataset, get_transforms
from src.models.cnn import build_cnn, build_cnn_no_bn, build_cnn_no_dropout
from src.utils import (
    CFG,
    classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
    set_all_seeds,
    worker_init_fn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "cnn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training helpers  (identical pattern used across all training scripts)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """Single training epoch — returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    """Validation/test epoch — returns (avg_loss, acc, preds, labels, probs)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels, all_probs


def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 criterion, device, epochs, patience, checkpoint_path, model_name):
    """
    Generic training loop with early stopping.

    Returns
    -------
    history : dict  keys: train_loss, val_loss, train_acc, val_acc
    """
    best_val_loss = float("inf")
    pat_count = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        # Cosine scheduler steps once per epoch
        if scheduler:
            scheduler.step()

        print(
            f"[{model_name}] Ep {epoch+1:3d}/{epochs} | "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pat_count = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"           -> OK Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch+1}.")
                break

    return history


# ---------------------------------------------------------------------------
# Shared helper: build DataLoaders
# ---------------------------------------------------------------------------

def make_loaders(train_csv, val_csv, test_csv, train_tf, val_tf, test_tf):
    """Create PotholeDataset instances and wrap them in DataLoaders."""
    train_ds = PotholeDataset(train_csv, transform=train_tf)
    val_ds   = PotholeDataset(val_csv,   transform=val_tf)
    test_ds  = PotholeDataset(test_csv,  transform=test_tf)

    bs = CFG.train.batch_size
    nw = CFG.data.num_workers
    pm = CFG.data.pin_memory

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pm,
                              worker_init_fn=worker_init_fn)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pm)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pm)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Shared helper: class-weighted CrossEntropyLoss
# ---------------------------------------------------------------------------

def make_criterion(labels_array, label_smoothing=0.0):
    """
    Build a CrossEntropyLoss weighted by inverse class frequency.

    label_smoothing >0 reduces over-confident predictions and can help
    generalisation when the training set is small.
    """
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_array),
        y=labels_array,
    )
    weight_tensor = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# Single run helper: train + evaluate + save artifacts
# ---------------------------------------------------------------------------

def single_run(
    run_name,
    model_fn,
    train_loader,
    val_loader,
    test_loader,
    train_labels,
    ckpt_dir,
    metrics_dir,
    plots_dir,
    epochs,
    patience,
    lr,
    weight_decay,
    label_smoothing,
    use_scheduler=True,
):
    """
    One complete train->validate->test cycle.

    Parameters mirror the training hyper-parameters and are explicit
    so ablations can selectively override them.
    """
    set_all_seeds(CFG.seed)  # reset seed so each ablation is comparable

    model = model_fn().to(DEVICE)
    criterion = make_criterion(train_labels, label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler:
        # CosineAnnealingLR decays LR from lr -> ~0 over T_max epochs;
        # avoids a fixed step schedule and tends to escape local minima.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

    ckpt_path = ckpt_dir / f"{run_name}_best.pth"

    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=DEVICE,
        epochs=epochs,
        patience=patience,
        checkpoint_path=ckpt_path,
        model_name=run_name,
    )

    # --- Save training curves ---------------------------------------------
    plot_path = plots_dir / f"{run_name}_training.png"
    plot_training_curves(history, model_name=run_name, save_path=plot_path)
    print(f"\n[{run_name}] Training curves -> {plot_path}")

    # --- Test evaluation with best checkpoint -----------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, test_acc, test_preds, test_labels_list, test_probs = eval_epoch(
        model, test_loader, criterion, DEVICE
    )

    test_metrics = classification_metrics(
        y_true=test_labels_list,
        y_pred=test_preds,
        y_prob=test_probs,
        model_name=run_name,
    )
    metrics_path = metrics_dir / f"{run_name}_test_metrics.json"
    save_metrics(test_metrics, metrics_path)
    print(f"[{run_name}] Metrics        -> {metrics_path}")

    # --- Confusion matrix -------------------------------------------------
    cm_path = plots_dir / f"{run_name}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true=test_labels_list,
        y_pred=test_preds,
        model_name=run_name,
        save_path=cm_path,
    )
    print(f"[{run_name}] Confusion mat  -> {cm_path}")

    # --- Print key metrics ------------------------------------------------
    print(f"\n{'-'*55}")
    print(f"  [{run_name}] TEST RESULTS")
    print(f"{'-'*55}")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
    print(f"{'-'*55}\n")

    return test_metrics


# ---------------------------------------------------------------------------
# Overfit sanity check
# ---------------------------------------------------------------------------

def overfit_sanity_check(train_ds, val_loader, ckpt_dir, plots_dir):
    """
    Train on a tiny 50-image subset for 20 epochs.

    If the model cannot reach ~100% training accuracy on 50 samples,
    there is likely a data-loading or forward-pass bug.
    No early stopping; we want to see the full overfit trajectory.
    """
    print("\n" + "="*55)
    print("  OVERFIT SANITY CHECK (50 images, 20 epochs)")
    print("="*55)

    set_all_seeds(CFG.seed)

    # Take first 50 samples from train set
    subset_indices = list(range(min(50, len(train_ds))))
    subset = Subset(train_ds, subset_indices)
    loader = DataLoader(subset, batch_size=16, shuffle=True,
                        num_workers=0, worker_init_fn=worker_init_fn)

    model = build_cnn().to(DEVICE)
    # No class weighting, no label smoothing — keep it simple for the sanity check
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.cnn.lr)

    history = {"train_loss": [], "train_acc": []}
    for epoch in range(20):
        tr_loss, tr_acc = train_epoch(model, loader, optimizer, criterion, DEVICE)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        print(f"  Ep {epoch+1:2d}/20 | tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}")

    final_acc = history["train_acc"][-1]
    if final_acc >= 0.95:
        print(f"\n  OK Sanity check PASSED (final train acc = {final_acc:.4f})")
    else:
        print(f"\n  ✗ Sanity check WARNING — train acc only {final_acc:.4f} (expected ≥ 0.95)")
        print("    Check data loading and model forward pass.")

    # Save the sanity-check curves for the report
    sc_plot = plots_dir / "cnn_overfit_sanity.png"
    plot_training_curves(
        {"train_loss": history["train_loss"],
         "val_loss":   history["train_loss"],    # dummy: same line
         "train_acc":  history["train_acc"],
         "val_acc":    history["train_acc"]},    # dummy
        model_name="cnn_overfit_sanity",
        save_path=sc_plot,
    )
    print(f"  Sanity-check plot -> {sc_plot}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_all_seeds(CFG.seed)

    print(f"\n{'='*60}")
    print(f"  PotholeIQ — {MODEL_NAME.upper()} training  (device: {DEVICE})")
    print(f"{'='*60}\n")

    # --- Paths ---------------------------------------------------------------
    splits_dir  = ROOT / CFG.paths.splits_dir
    ckpt_dir    = ROOT / CFG.paths.checkpoints_dir
    metrics_dir = ROOT / CFG.paths.metrics_dir
    plots_dir   = ROOT / CFG.paths.plots_dir

    for d in (ckpt_dir, metrics_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train.csv"
    val_csv   = splits_dir / "val.csv"
    test_csv  = splits_dir / "test.csv"

    for csv_path in (train_csv, val_csv, test_csv):
        if not csv_path.exists():
            print(f"[ERROR] Split file not found: {csv_path}")
            sys.exit(1)

    # --- Transforms ----------------------------------------------------------
    train_tf_aug = get_transforms(mode="train")   # augmented  (flips, colour jitter, etc.)
    train_tf_val = get_transforms(mode="val")     # plain      (resize + normalise only)
    val_tf       = get_transforms(mode="val")
    test_tf      = get_transforms(mode="val")

    # --- Main loaders (augmented train) -------------------------------------
    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader) = make_loaders(
        train_csv, val_csv, test_csv,
        train_tf_aug, val_tf, test_tf,
    )

    train_labels = np.array(train_ds.labels)
    print(f"Dataset sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"Label distribution (train): {dict(zip(*np.unique(train_labels, return_counts=True)))}\n")

    # =========================================================================
    # 1. Main CNN run
    # =========================================================================
    print("\n" + "="*55)
    print("  RUN 1/4 — Full CNN (BN + Dropout + Augmentation)")
    print("="*55)
    single_run(
        run_name="cnn",
        model_fn=build_cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_labels=train_labels,
        ckpt_dir=ckpt_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
        epochs=CFG.cnn.epochs,
        patience=CFG.train.early_stopping_patience,
        lr=CFG.cnn.lr,
        weight_decay=CFG.cnn.weight_decay,
        label_smoothing=float(CFG.cnn.label_smoothing),
        use_scheduler=True,
    )

    # =========================================================================
    # 2. Ablation: No Augmentation
    # =========================================================================
    print("\n" + "="*55)
    print("  RUN 2/4 — Ablation: No Augmentation")
    print("="*55)

    # Use val-style transforms for train — identical preprocessing, no random flips/jitter
    (train_ds_na, _, _,
     train_loader_na, _, _) = make_loaders(
        train_csv, val_csv, test_csv,
        train_tf_val, val_tf, test_tf,
    )

    single_run(
        run_name="cnn_no_aug",
        model_fn=build_cnn,
        train_loader=train_loader_na,
        val_loader=val_loader,
        test_loader=test_loader,
        train_labels=np.array(train_ds_na.labels),
        ckpt_dir=ckpt_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
        epochs=CFG.cnn.epochs,
        patience=CFG.train.early_stopping_patience,
        lr=CFG.cnn.lr,
        weight_decay=CFG.cnn.weight_decay,
        label_smoothing=float(CFG.cnn.label_smoothing),
        use_scheduler=True,
    )

    # =========================================================================
    # 3. Ablation: No BatchNorm
    # =========================================================================
    print("\n" + "="*55)
    print("  RUN 3/4 — Ablation: No BatchNorm")
    print("="*55)
    single_run(
        run_name="cnn_no_bn",
        model_fn=build_cnn_no_bn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_labels=train_labels,
        ckpt_dir=ckpt_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
        epochs=CFG.cnn.epochs,
        patience=CFG.train.early_stopping_patience,
        lr=CFG.cnn.lr,
        weight_decay=CFG.cnn.weight_decay,
        label_smoothing=float(CFG.cnn.label_smoothing),
        use_scheduler=True,
    )

    # =========================================================================
    # 4. Ablation: No Dropout
    # =========================================================================
    print("\n" + "="*55)
    print("  RUN 4/4 — Ablation: No Dropout")
    print("="*55)
    single_run(
        run_name="cnn_no_dropout",
        model_fn=build_cnn_no_dropout,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_labels=train_labels,
        ckpt_dir=ckpt_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
        epochs=CFG.cnn.epochs,
        patience=CFG.train.early_stopping_patience,
        lr=CFG.cnn.lr,
        weight_decay=CFG.cnn.weight_decay,
        label_smoothing=float(CFG.cnn.label_smoothing),
        use_scheduler=True,
    )

    # =========================================================================
    # 5. Overfit sanity check
    # =========================================================================
    overfit_sanity_check(train_ds, val_loader, ckpt_dir, plots_dir)

    print("\nOK All CNN runs complete.\n")


if __name__ == "__main__":
    main()
