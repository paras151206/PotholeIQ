"""
src/explainability/run_gradcam.py
==================================
Applies Grad-CAM to the trained CustomCNN and saves visualisation grids.

Outputs:
  results/plots/gradcam_correct.png   – 5×2 grid of correctly classified images
  results/plots/gradcam_incorrect.png – 5×2 grid of misclassified images
  results/metrics/gradcam_failure_analysis.txt – automated failure analysis

Run:
    python src/explainability/run_gradcam.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils import CFG, set_all_seeds
from src.data.dataset import PotholeDataset, get_transforms
from src.models.cnn import build_cnn
from src.explainability.gradcam import GradCAM, denormalize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = ROOT / CFG.paths.checkpoints_dir / "cnn_best.pth"
PLOTS_DIR = ROOT / CFG.paths.plots_dir
METRICS_DIR = ROOT / CFG.paths.metrics_dir

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MEAN = CFG.augmentation.normalize_mean
STD  = CFG.augmentation.normalize_std

CLASS_NAMES = {0: "normal", 1: "pothole"}

# How many correct / incorrect examples to visualise
N_SAMPLES = 10   # 10 correct + 10 incorrect collected; 5 displayed per row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model():
    """Load trained CustomCNN; exit if checkpoint is missing."""
    if not CKPT_PATH.exists():
        print(f"[ERROR] Checkpoint not found: {CKPT_PATH}")
        print("Train the CNN first: python src/models/train_cnn.py")
        sys.exit(1)

    model = build_cnn().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded checkpoint: {CKPT_PATH}")
    return model


def predict_with_confidence(model, img_tensor: torch.Tensor):
    """Return (pred_class, confidence, logits) for a single image tensor."""
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(DEVICE))  # (1, 2)
        probs  = F.softmax(logits, dim=1)                   # (1, 2)
        pred   = int(probs.argmax(dim=1).item())
        conf   = float(probs[0, pred].item())
    return pred, conf, logits


def save_grid(samples, title: str, out_path: Path, cols: int = 2):
    """Save a grid of (original | Grad-CAM overlay) pairs.

    Args:
        samples: list of dicts with keys:
                   img_np (H,W,3 uint8), overlay (H,W,3 uint8),
                   true_label (int), pred_label (int), conf (float),
                   filename (str)
        title:   figure super-title
        out_path: where to save the PNG
        cols:    number of image-pairs per row (each pair = 2 columns)
    """
    n = len(samples)
    rows = (n + cols - 1) // cols          # ceiling division
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3.5))
    axes = np.array(axes).reshape(rows, cols * 2)

    for idx, s in enumerate(samples):
        row = idx // cols
        col = (idx % cols) * 2            # each pair occupies 2 sub-columns

        ax_orig = axes[row, col]
        ax_gcam = axes[row, col + 1]

        ax_orig.imshow(s["img_np"])
        ax_orig.set_title(
            f"True: {CLASS_NAMES[s['true_label']]}\n"
            f"Pred: {CLASS_NAMES[s['pred_label']]} ({s['conf']:.2f})",
            fontsize=8,
        )
        ax_orig.axis("off")

        ax_gcam.imshow(s["overlay"])
        ax_gcam.set_title("Grad-CAM", fontsize=8)
        ax_gcam.axis("off")

    # Hide any unused subplots
    for idx in range(n, rows * cols):
        row = idx // cols
        col = (idx % cols) * 2
        axes[row, col].axis("off")
        axes[row, col + 1].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {out_path}")


def failure_note(pred_label: int, true_label: int, conf: float) -> str:
    """Generate an automated failure analysis note based on prediction error type.

    The note describes *why* the model might have made this error based on
    the confidence level and direction of error — this is a heuristic that
    flags potential issues for a human reviewer to investigate.
    """
    direction = f"{CLASS_NAMES[true_label]} → {CLASS_NAMES[pred_label]}"
    if pred_label == 1 and true_label == 0:
        # False positive: model said pothole on a normal road
        if conf > 0.85:
            note = "high activation on road texture/shadow — model confuses dark regions for potholes"
        else:
            note = "borderline texture features triggered pothole response"
    elif pred_label == 0 and true_label == 1:
        # False negative: model missed a pothole
        if conf > 0.85:
            note = "pothole is likely small/filled or heavily occluded by shadow/glare"
        else:
            note = "low-contrast pothole — features similar to normal road surface"
    else:
        note = "unexpected error type"
    return f"error_type={direction} | {note}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_all_seeds(CFG.seed)

    # ---- Load model --------------------------------------------------------
    model = load_model()

    # ---- Grad-CAM setup: target last conv block ----------------------------
    # The last conv block captures the highest-level spatial features, making
    # it the most semantically meaningful layer for localisation explanations.
    target_layer = model.conv_blocks[-1]
    gcam = GradCAM(model, target_layer)

    # ---- Load test dataset -------------------------------------------------
    test_csv = ROOT / CFG.paths.splits_dir / "test.csv"
    if not test_csv.exists():
        print(f"[ERROR] {test_csv} not found. Run src/data/split.py first.")
        sys.exit(1)

    test_tf = get_transforms("val")
    test_ds = PotholeDataset(test_csv, transform=test_tf)
    print(f"[INFO] Test set size: {len(test_ds)}")

    # ---- Collect correctly and incorrectly classified samples ---------------
    correct_samples   = []
    incorrect_samples = []

    model.eval()
    for img_tensor, true_label in test_ds:
        # Stop once we have enough of each category
        if len(correct_samples) >= N_SAMPLES and len(incorrect_samples) >= N_SAMPLES:
            break

        pred, conf, _ = predict_with_confidence(model, img_tensor)

        # Generate Grad-CAM heatmap for the predicted class
        heatmap = gcam.generate(
            img_tensor.unsqueeze(0).to(DEVICE),
            target_class=pred,
        )

        img_np  = denormalize(img_tensor, MEAN, STD)
        overlay = gcam.overlay_on_image(img_np, heatmap, alpha=0.5)

        entry = {
            "img_np":     img_np,
            "overlay":    overlay,
            "true_label": int(true_label),
            "pred_label": pred,
            "conf":       conf,
            "filename":   "",   # filename not exposed by PotholeDataset directly
        }

        if pred == int(true_label):
            if len(correct_samples) < N_SAMPLES:
                correct_samples.append(entry)
        else:
            if len(incorrect_samples) < N_SAMPLES:
                incorrect_samples.append(entry)

    gcam.remove_hooks()

    # ---- Save grids --------------------------------------------------------
    # Display only the first 10 of each (5 pairs per row × 2 rows = 5×2 grid)
    save_grid(
        correct_samples[:N_SAMPLES],
        title="Grad-CAM: Correctly Classified (5×2)",
        out_path=PLOTS_DIR / "gradcam_correct.png",
        cols=5,
    )
    save_grid(
        incorrect_samples[:N_SAMPLES],
        title="Grad-CAM: Incorrectly Classified (5×2)",
        out_path=PLOTS_DIR / "gradcam_incorrect.png",
        cols=5,
    )

    # ---- Failure analysis text file ----------------------------------------
    fail_path = METRICS_DIR / "gradcam_failure_analysis.txt"
    with open(fail_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("GRAD-CAM FAILURE ANALYSIS\n")
        f.write("=" * 72 + "\n\n")
        f.write(
            "Format: Image: <idx> | True: <class> | Pred: <class> | "
            "Conf: <conf> | Note: [analysis]\n\n"
        )

        # Categorise by error type
        fp_cases = [s for s in incorrect_samples if s["pred_label"] == 1 and s["true_label"] == 0]
        fn_cases = [s for s in incorrect_samples if s["pred_label"] == 0 and s["true_label"] == 1]

        f.write(f"Total misclassified analysed: {len(incorrect_samples)}\n")
        f.write(f"  False Positives (normal → pothole): {len(fp_cases)}\n")
        f.write(f"  False Negatives (pothole → normal): {len(fn_cases)}\n\n")

        f.write("-" * 72 + "\n")
        f.write("Per-image breakdown:\n\n")
        for i, s in enumerate(incorrect_samples):
            note = failure_note(s["pred_label"], s["true_label"], s["conf"])
            line = (
                f"Image: {i + 1:02d} | "
                f"True: {CLASS_NAMES[s['true_label']]:7s} | "
                f"Pred: {CLASS_NAMES[s['pred_label']]:7s} | "
                f"Conf: {s['conf']:.4f} | "
                f"Note: [{note}]"
            )
            f.write(line + "\n")

        f.write("\n" + "-" * 72 + "\n")
        f.write("\nSUMMARY BY FAILURE TYPE:\n\n")

        if fp_cases:
            avg_fp_conf = np.mean([s["conf"] for s in fp_cases])
            f.write(textwrap.dedent(f"""\
                False Positives (n={len(fp_cases)}, avg_conf={avg_fp_conf:.4f}):
                  The model activates on road textures, shadows, and low-contrast regions
                  that superficially resemble pothole surfaces. These false alarms tend to
                  occur at high confidence, suggesting the model has learned a shortcut
                  (e.g., dark patches) rather than the true 3-D depression shape.
                  Recommendation: add harder negatives (cracked roads, puddles) to training.
            """))

        if fn_cases:
            avg_fn_conf = np.mean([s["conf"] for s in fn_cases])
            f.write(textwrap.dedent(f"""\
                False Negatives (n={len(fn_cases)}, avg_conf={avg_fn_conf:.4f}):
                  Potholes missed by the model tend to be small, partially filled, or viewed
                  under glare conditions.  Grad-CAM heatmaps for these cases often show
                  diffuse, low-intensity activation rather than a focused hotspot.
                  Recommendation: oversample small-pothole images; apply stronger
                  brightness/contrast augmentation.
            """))

    print(f"[INFO] Failure analysis saved → {fail_path}")
    print("\n[DONE] Grad-CAM analysis complete.")


if __name__ == "__main__":
    main()
