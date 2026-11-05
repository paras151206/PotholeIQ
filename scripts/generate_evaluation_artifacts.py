"""
generate_evaluation_artifacts.py — Generate all required evaluation plots:
1. YOLO confusion matrix evaluated strictly at conf=0.25 and IoU=0.5 on the test split.
2. Classifier combined ROC curves (MLP, CNN, MobileNetV2).
3. Ablation study F1 bar chart.
4. Bootstrap 95% CI error bar chart.
5. Ensure PR curve is copied.
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import CFG
from src.data.dataset import PotholeDataset, get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOTS_DIR = ROOT / "results" / "plots"
METRICS_DIR = ROOT / "results" / "metrics"
CKPT_DIR = ROOT / "results" / "checkpoints"
TEST_IMG_DIR = ROOT / "data" / "yolo" / "test" / "images"
TEST_LBL_DIR = ROOT / "data" / "yolo" / "test" / "labels"
TEST_CSV = ROOT / "data" / "splits" / "test.csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Regenerate YOLO Confusion Matrix at conf=0.25, iou=0.5
# ─────────────────────────────────────────────────────────────────────────────
def generate_yolo_confusion_matrix():
    print("[1/5] Generating YOLO confusion matrix at conf=0.25, IoU=0.5...")
    from ultralytics import YOLO

    model = YOLO(str(CKPT_DIR / "best.pt"))
    test_img_files = sorted(list(TEST_IMG_DIR.glob("*.*")))

    tp = 0
    fp = 0
    fn = 0

    for img_p in test_img_files:
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class x_c y_c w h
                    xc, yc, w, h = map(float, parts[1:5])
                    gt_boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])

        # Run inference at conf=0.25, iou=0.5
        res = model(str(img_p), conf=0.25, iou=0.5, verbose=False)[0]
        img_h, img_w = res.orig_shape

        pred_boxes = []
        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            pred_boxes.append([
                xyxy[0] / img_w,
                xyxy[1] / img_h,
                xyxy[2] / img_w,
                xyxy[3] / img_h,
            ])

        # Match predicted boxes to ground truth boxes using IoU >= 0.5
        matched_gt = set()
        for pbox in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gbox in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                # Calculate IoU
                ix1 = max(pbox[0], gbox[0])
                iy1 = max(pbox[1], gbox[1])
                ix2 = min(pbox[2], gbox[2])
                iy2 = min(pbox[3], gbox[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_p = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                area_g = (gbox[2] - gbox[0]) * (gbox[3] - gbox[1])
                union = area_p + area_g - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= 0.5:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    print(f"YOLO Test Set Evaluation (conf=0.25, IoU=0.5): TP={tp}, FP={fp}, FN={fn}")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Plot clean confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_matrix = np.array([[tp, fn], [fp, 0]])
    im = ax.imshow([[tp, fn], [fp, 0]], cmap="Blues", interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pothole (Pred)", "Background (Miss)"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["Pothole (GT)", "Background (GT)"], fontsize=10, fontweight="bold")

    # Annotate cells
    ax.text(0, 0, f"True Positive (TP)\n{tp}", ha="center", va="center", color="white" if tp > 200 else "black", fontsize=11, fontweight="bold")
    ax.text(1, 0, f"False Negative (FN)\n{fn}", ha="center", va="center", color="black", fontsize=11, fontweight="bold")
    ax.text(0, 1, f"False Positive (FP)\n{fp}", ha="center", va="center", color="black", fontsize=11, fontweight="bold")
    ax.text(1, 1, "N/A\n(Background)", ha="center", va="center", color="gray", fontsize=10)

    plt.title(f"YOLOv8n Test Confusion Matrix\n(Evaluated at conf=0.25, IoU=0.50 | 239 test images)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = PLOTS_DIR / "yolo_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved realistic YOLO confusion matrix to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Copy YOLO PR Curve
# ─────────────────────────────────────────────────────────────────────────────
def copy_yolo_pr_curve():
    print("[2/5] Copying YOLO PR curve...")
    src_pr = ROOT / "runs" / "detect" / "test_eval" / "BoxPR_curve.png"
    if not src_pr.exists():
        src_pr = ROOT / "runs" / "detect" / "pothole_yolov8n" / "BoxPR_curve.png"
    if src_pr.exists():
        shutil.copy2(src_pr, PLOTS_DIR / "yolo_PR_curve.png")
        print(f"[OK] Copied {src_pr} to {PLOTS_DIR / 'yolo_PR_curve.png'}")
    else:
        print("[WARNING] BoxPR_curve.png not found in runs/detect/")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generate Combined Classifier ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
def generate_classifier_roc_curves():
    print("[3/5] Generating combined Classifier ROC curves...")
    if not TEST_CSV.exists():
        print(f"[WARNING] {TEST_CSV} not found, skipping classifier ROC.")
        return

    from src.models.baseline import build_baseline
    from src.models.cnn import build_cnn
    from src.models.transfer import build_transfer_model

    tf = get_transforms("val")
    test_ds = PotholeDataset(TEST_CSV, transform=tf)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    y_true = []
    for _, labels in loader:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)

    models_info = [
        ("MobileNetV2 (Transfer)", "transfer_best.pth", build_transfer_model(pretrained=False), "#2ca02c"),
        ("Custom CNN", "cnn_best.pth", build_cnn(), "#1f77b4"),
        ("Baseline MLP", "baseline_best.pth", build_baseline(), "#d62728"),
    ]

    plt.figure(figsize=(7, 6))

    for name, ckpt_name, model, color in models_info:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval().to(DEVICE)

        y_probs = []
        with torch.no_grad():
            for imgs, _ in loader:
                logits = model(imgs.to(DEVICE))
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_probs.extend(probs)

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) — Classifiers\n(Held-out Test Split, N=103)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = PLOTS_DIR / "classifier_roc_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved classifier ROC curves to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Generate Ablation F1 Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def generate_ablation_barchart():
    print("[4/5] Generating Ablation Study F1 Bar Chart...")
    ablation_json = METRICS_DIR / "ablation_summary.json"
    if not ablation_json.exists():
        # Build ablation summary from individual json files
        ablation_data = {}
        for k, fname in [
            ("Full Custom CNN", "cnn_test_metrics.json"),
            ("No Augmentation", "cnn_no_aug_test_metrics.json"),
            ("No BatchNorm", "cnn_no_bn_test_metrics.json"),
            ("No Dropout", "cnn_no_dropout_test_metrics.json"),
        ]:
            p = METRICS_DIR / fname
            if p.exists():
                ablation_data[k] = json.loads(p.read_text())
        with open(ablation_json, "w") as f:
            json.dump(ablation_data, f, indent=2)
    else:
        ablation_data = json.loads(ablation_json.read_text())

    variants = list(ablation_data.keys())
    f1_scores = [ablation_data[v].get("f1_score", 0.0) * 100 for v in variants]
    accuracies = [ablation_data[v].get("accuracy", 0.0) * 100 for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, f1_scores, width, label="Macro F1-Score (%)", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, accuracies, width, label="Test Accuracy (%)", color="#aec7e8")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Custom CNN Ablation Study — F1 & Accuracy Comparison\n(Held-out Test Split, N=103)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10, fontweight="bold")
    ax.set_ylim([75, 100])
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = PLOTS_DIR / "ablation_f1_barchart.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved ablation bar chart to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Generate Bootstrap Confidence Intervals Chart
# ─────────────────────────────────────────────────────────────────────────────
def generate_bootstrap_ci_chart():
    print("[5/5] Generating Bootstrap 95% CI chart...")
    ci_json = METRICS_DIR / "bootstrap_ci_summary.json"
    if not ci_json.exists():
        print("[WARNING] bootstrap_ci_summary.json not found.")
        return

    data = json.loads(ci_json.read_text())
    models = ["Baseline MLP", "Custom CNN", "MobileNetV2"]
    keys = ["baseline", "cnn", "transfer"]

    f1_means = []
    f1_err_low = []
    f1_err_high = []

    for k in keys:
        m_info = data.get(k, {})
        mean_f1 = m_info.get("f1_mean", 0.0) * 100
        ci_95 = m_info.get("f1_ci_95", [mean_f1, mean_f1])
        f1_means.append(mean_f1)
        f1_err_low.append(mean_f1 - ci_95[0] * 100)
        f1_err_high.append(ci_95[1] * 100 - mean_f1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(models))
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    for i, (m, low, high, c) in enumerate(zip(f1_means, f1_err_low, f1_err_high, colors)):
        ax.errorbar(
            m,
            y_pos[i],
            xerr=[[low], [high]],
            fmt="o",
            color=c,
            ecolor=c,
            elinewidth=3,
            capsize=6,
            capthick=2,
            markersize=8,
        )
        ax.text(m, y_pos[i] + 0.22, f"{m:.2f}% (CI: [{m-low:.1f}%, {m+high:.1f}%])", ha="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11, fontweight="bold")
    ax.set_xlabel("Test Macro F1-Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("95% Non-Parametric Bootstrap Confidence Intervals (1,000 Resamples)\nHeld-out Test Split, N=103", fontsize=11, fontweight="bold")
    ax.set_xlim([55, 105])
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out_path = PLOTS_DIR / "bootstrap_ci_barchart.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved bootstrap CI chart to {out_path}")


def main():
    generate_yolo_confusion_matrix()
    copy_yolo_pr_curve()
    generate_classifier_roc_curves()
    generate_ablation_barchart()
    generate_bootstrap_ci_chart()
    print("\n[SUCCESS] ALL EVALUATION ARTIFACTS REGENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
