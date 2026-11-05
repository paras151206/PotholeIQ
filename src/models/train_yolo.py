"""
train_yolo.py — Fine-tune YOLOv8n on Kaggle Pothole Dataset and evaluate on TEST split.
Exports test metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95), PR curve,
confusion matrix, and 8 test prediction visualizations.
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"
TEST_IMG_DIR = ROOT / "data" / "yolo" / "test" / "images"
CKPT_DIR = ROOT / "results" / "checkpoints"
PLOTS_DIR = ROOT / "results" / "plots"
METRICS_DIR = ROOT / "results" / "metrics"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        print(f"[ERROR] {DATA_YAML} not found. Run src/data/prepare_yolo_dataset.py first.")
        sys.exit(1)

    print("=======================================================")
    print("  PotholeIQ YOLOv8n Fine-Tuning & Evaluation")
    print("=======================================================")

    # Load COCO-pretrained YOLOv8n
    print("[1/4] Loading pretrained YOLOv8n...")
    model = YOLO("yolov8n.pt")

    # Training parameters
    # imgsz=416, epochs=8 for high quality CPU training within time budget
    epochs = 8
    imgsz = 416
    batch = 16

    print(f"[2/4] Starting fine-tuning: epochs={epochs}, imgsz={imgsz}, batch={batch}...")
    train_results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        project=str(ROOT / "runs" / "detect"),
        name="pothole_yolov8n",
        exist_ok=True,
        seed=42,
        plots=True,
        verbose=True,
    )

    # Path to trained best.pt
    trained_best_pt = ROOT / "runs" / "detect" / "pothole_yolov8n" / "weights" / "best.pt"
    if not trained_best_pt.exists():
        # fallback to last.pt if early stopping or direct run
        trained_best_pt = ROOT / "runs" / "detect" / "pothole_yolov8n" / "weights" / "last.pt"

    print(f"[INFO] Best weights saved at: {trained_best_pt}")

    # Copy best.pt to results/checkpoints/best.pt and yolov8n_best.pt
    dest_best = CKPT_DIR / "best.pt"
    shutil.copy2(trained_best_pt, dest_best)
    shutil.copy2(trained_best_pt, CKPT_DIR / "yolov8n_best.pt")
    print(f"[INFO] Saved checkpoint to {dest_best} ({dest_best.stat().st_size / (1024*1024):.2f} MB)")

    # [3/4] Evaluate on HELD-OUT TEST SPLIT ONLY
    print("\n[3/4] Evaluating strictly on held-out TEST split (data/yolo/test)...")
    best_model = YOLO(str(dest_best))
    test_metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=imgsz,
        batch=batch,
        plots=True,
        project=str(ROOT / "runs" / "detect"),
        name="test_eval",
        exist_ok=True,
    )

    # Extract metrics
    # test_metrics.box.mp = Mean Precision
    # test_metrics.box.mr = Mean Recall
    # test_metrics.box.map50 = mAP@0.5
    # test_metrics.box.map = mAP@0.5:0.95
    precision = float(test_metrics.box.mp)
    recall = float(test_metrics.box.mr)
    map50 = float(test_metrics.box.map50)
    map50_95 = float(test_metrics.box.map)

    results_dict = {
        "model": "YOLOv8n",
        "dataset": "Kaggle Potholes Detection YOLO (1590 images, 5684 boxes)",
        "test_split_size": len(list(TEST_IMG_DIR.glob("*.*"))),
        "imgsz": imgsz,
        "epochs": epochs,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map_50": round(map50, 4),
        "map_50_95": round(map50_95, 4),
    }

    metrics_file = METRICS_DIR / "yolo_test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[INFO] Test metrics saved to {metrics_file}")

    # Copy curves from run directory to results/plots/
    run_dir = ROOT / "runs" / "detect" / "pothole_yolov8n"
    test_eval_dir = ROOT / "runs" / "detect" / "test_eval"

    for src_dir in [test_eval_dir, run_dir]:
        if src_dir.exists():
            for p_name in ["PR_curve.png", "confusion_matrix.png", "results.png", "F1_curve.png"]:
                src_file = src_dir / p_name
                if src_file.exists():
                    shutil.copy2(src_file, PLOTS_DIR / f"yolo_{p_name}")
                    print(f"[INFO] Copied {p_name} to results/plots/yolo_{p_name}")

    # [4/4] Generate 8 example test prediction overlays
    print("\n[4/4] Generating 8 test prediction visualizations for results/plots/...")
    test_images = sorted(list(TEST_IMG_DIR.glob("*.*")))[:8]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, img_path in enumerate(test_images):
        pred_res = best_model(str(img_path), conf=0.25, verbose=False)[0]
        # pred_res.plot() returns BGR numpy array with detections
        annotated_bgr = pred_res.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(annotated_rgb)
        axes[idx].set_title(f"Test #{idx+1} ({len(pred_res.boxes)} found)", fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle("YOLOv8n Held-Out Test Set Predictions (Pothole Detection)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    gallery_path = PLOTS_DIR / "yolo_test_predictions_gallery.png"
    plt.savefig(gallery_path, dpi=150)
    plt.close()
    print(f"[INFO] 8 test prediction samples saved to {gallery_path}")

    print("\n================ YOLO TEST EVALUATION SUMMARY ================")
    print(f"  Model:            YOLOv8n (pretrained on COCO, fine-tuned)")
    print(f"  Test Images:      {results_dict['test_split_size']}")
    print(f"  Precision:        {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:           {recall:.4f} ({recall*100:.2f}%)")
    print(f"  mAP@0.5:          {map50:.4f} ({map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95:     {map50_95:.4f} ({map50_95*100:.2f}%)")
    print("==============================================================")


if __name__ == "__main__":
    main()

