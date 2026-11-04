"""
train_baseline.py — Training script for the TinyMLP baseline model (Module M2)

Usage:
    python src/models/train_baseline.py

What this script does:
1.  Loads train / val / test splits from data/splits/*.csv
2.  Builds DataLoaders with standard (non-augmented) transforms
3.  Constructs TinyMLP via build_baseline()
4.  Computes class weights to handle label imbalance (potholes may be rare)
5.  Trains with Adam + CrossEntropyLoss (weighted) for CFG.baseline.epochs
6.  Applies early stopping based on validation loss
7.  Evaluates the best checkpoint on the held-out test set exactly once
8.  Saves:
        results/checkpoints/baseline_best.pth
        results/metrics/baseline_test_metrics.json
        results/plots/baseline_training.png
        results/plots/baseline_confusion_matrix.png
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.data.dataset import PotholeDataset, get_transforms
from src.models.baseline import build_baseline
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
MODEL_NAME = "baseline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """
    One full pass over the training set.

    Returns
    -------
    avg_loss : float
    accuracy : float  (fraction of correct predictions)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)              # (B, 2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    """
    Evaluate model on a DataLoader without updating weights.

    Returns
    -------
    avg_loss : float
    accuracy : float
    preds    : list[int]   predicted class indices
    labels   : list[int]   ground-truth class indices
    probs    : list[float] predicted probability for the positive class (pothole)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]   # pothole probability

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
    Generic training loop with early stopping and best-checkpoint saving.

    Parameters
    ----------
    model           : nn.Module
    train_loader    : DataLoader
    val_loader      : DataLoader
    optimizer       : torch.optim.Optimizer
    scheduler       : LR scheduler or None
    criterion       : loss function
    device          : torch.device
    epochs          : int   maximum number of epochs
    patience        : int   early-stopping patience
    checkpoint_path : Path  where to save the best model weights
    model_name      : str   used for printing

    Returns
    -------
    history : dict  with keys train_loss, val_loss, train_acc, val_acc
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

        # Step scheduler if one is provided
        if scheduler:
            scheduler.step()

        print(
            f"[{model_name}] Ep {epoch+1:3d}/{epochs} | "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pat_count = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"           -> OK Best checkpoint saved (val_loss={val_loss:.4f})")
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"[{model_name}] Early stopping triggered at epoch {epoch+1}.")
                break

    return history


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # --- Reproducibility ----------------------------------------------------
    set_all_seeds(CFG.seed)
    print(f"\n{'='*60}")
    print(f"  PotholeIQ — {MODEL_NAME.upper()} training")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}\n")

    # --- Resolve paths -------------------------------------------------------
    splits_dir  = ROOT / CFG.paths.splits_dir
    ckpt_dir    = ROOT / CFG.paths.checkpoints_dir
    metrics_dir = ROOT / CFG.paths.metrics_dir
    plots_dir   = ROOT / CFG.paths.plots_dir

    for d in (ckpt_dir, metrics_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train.csv"
    val_csv   = splits_dir / "val.csv"
    test_csv  = splits_dir / "test.csv"

    # Fail early if splits are missing so the error is obvious
    for csv_path in (train_csv, val_csv, test_csv):
        if not csv_path.exists():
            print(f"[ERROR] Split file not found: {csv_path}")
            print("        Run the data preparation module first.")
            sys.exit(1)

    # --- Transforms ----------------------------------------------------------
    # Baseline uses non-augmented transforms for both train and val.
    # The MLP has no spatial inductive bias, so heavy augmentation
    # would not help but could add noise.
    train_tf = get_transforms(mode="val")   # resize + normalise only
    val_tf   = get_transforms(mode="val")
    test_tf  = get_transforms(mode="val")

    # --- Datasets & DataLoaders ----------------------------------------------
    train_ds = PotholeDataset(train_csv, transform=train_tf)
    val_ds   = PotholeDataset(val_csv,   transform=val_tf)
    test_ds  = PotholeDataset(test_csv,  transform=test_tf)

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

    print(f"Dataset sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # --- Class weights (handle label imbalance) ------------------------------
    # compute_class_weight returns an array [w_class0, w_class1] such that
    # the rarer class receives higher weight during loss computation.
    labels_arr = np.array(train_ds.labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_arr),
        y=labels_arr,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {class_weights}")

    # --- Model ---------------------------------------------------------------
    model = build_baseline().to(DEVICE)
    print(f"\nModel architecture:\n{model}\n")

    # --- Loss, optimiser, (no scheduler for baseline) -----------------------
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.baseline.lr,
        weight_decay=CFG.baseline.weight_decay,
    )

    # --- Training ------------------------------------------------------------
    checkpoint_path = ckpt_dir / f"{MODEL_NAME}_best.pth"

    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,          # baseline keeps it simple: no LR schedule
        criterion=criterion,
        device=DEVICE,
        epochs=CFG.baseline.epochs,
        patience=CFG.train.early_stopping_patience,
        checkpoint_path=checkpoint_path,
        model_name=MODEL_NAME,
    )

    # --- Save training curves ------------------------------------------------
    plot_path = plots_dir / f"{MODEL_NAME}_training.png"
    plot_training_curves(history, model_name=MODEL_NAME, save_path=plot_path)
    print(f"\nTraining curves saved -> {plot_path}")

    # --- Test evaluation (run exactly ONCE with best checkpoint) -------------
    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    _, test_acc, test_preds, test_labels, test_probs = eval_epoch(
        model, test_loader, criterion, DEVICE
    )

    test_metrics = classification_metrics(
        y_true=test_labels,
        y_pred=test_preds,
        y_prob=test_probs,
        model_name=MODEL_NAME,
    )

    # --- Persist metrics -----------------------------------------------------
    metrics_path = metrics_dir / f"{MODEL_NAME}_test_metrics.json"
    save_metrics(test_metrics, metrics_path)
    print(f"Test metrics saved  -> {metrics_path}")

    # --- Confusion matrix ----------------------------------------------------
    cm_path = plots_dir / f"{MODEL_NAME}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true=test_labels,
        y_pred=test_preds,
        model_name=MODEL_NAME,
        save_path=cm_path,
    )
    print(f"Confusion matrix    -> {cm_path}")

    # --- Print summary -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  {MODEL_NAME.upper()} FINAL TEST RESULTS")
    print(f"{'='*60}")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:.4f}")
        else:
            print(f"  {k:<25} {v}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
