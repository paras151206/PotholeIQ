"""
prepare_yolo_dataset.py — Process raw YOLO pothole dataset into clean 70/15/15 splits,
verify bounding boxes, check for duplicate leakage, and save visual proof.
"""

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "yolo_raw"
OUT_DIR = ROOT / "data" / "yolo"
PLOTS_DIR = ROOT / "results" / "plots"
METRICS_DIR = ROOT / "results" / "metrics"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def compute_file_hash(filepath: Path) -> str:
    """MD5 checksum of file contents."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_boxes(label_path: Path) -> int:
    """Count number of bounding boxes in a YOLO label file."""
    if not label_path.exists():
        return 0
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    return len(lines)


def draw_yolo_boxes(image_path: Path, label_path: Path) -> np.ndarray:
    """Render ground-truth YOLO bounding boxes on the image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h

            x1 = int(max(0, x_center - box_w / 2))
            y1 = int(max(0, y_center - box_h / 2))
            x2 = int(min(w, x_center + box_w / 2))
            y2 = int(min(h, y_center + box_h / 2))

            # Draw green bounding box for ground truth
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                "pothole",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    print("[1/5] Collecting raw images and labels...")
    raw_images = []
    for split in ["train", "valid"]:
        img_dir = RAW_DIR / split / "images"
        lbl_dir = RAW_DIR / split / "labels"
        if not img_dir.exists():
            continue
        for img_p in img_dir.glob("*.*"):
            if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                lbl_p = lbl_dir / f"{img_p.stem}.txt"
                raw_images.append((img_p, lbl_p))

    print(f"Total raw pairs found: {len(raw_images)}")

    print("[2/5] Checking for duplicates and verifying labels...")
    seen_hashes = {}
    unique_pairs = []
    duplicates_removed = 0
    total_boxes = 0

    for img_p, lbl_p in raw_images:
        fhash = compute_file_hash(img_p)
        if fhash in seen_hashes:
            duplicates_removed += 1
            continue
        seen_hashes[fhash] = img_p
        num_b = count_boxes(lbl_p)
        total_boxes += num_b
        unique_pairs.append((img_p, lbl_p, num_b))

    print(f"Unique images: {len(unique_pairs)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Total ground-truth bounding boxes: {total_boxes}")
    print(f"Average boxes per image: {total_boxes / len(unique_pairs):.2f}")

    print("[3/5] Performing 70/15/15 deterministic split...")
    random.shuffle(unique_pairs)
    n = len(unique_pairs)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    splits = {
        "train": unique_pairs[:n_train],
        "val": unique_pairs[n_train : n_train + n_val],
        "test": unique_pairs[n_train + n_val :],
    }

    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    # Verify 0 leakage between sets
    train_hashes = {compute_file_hash(p[0]) for p in splits["train"]}
    val_hashes = {compute_file_hash(p[0]) for p in splits["val"]}
    test_hashes = {compute_file_hash(p[0]) for p in splits["test"]}
    assert len(train_hashes & val_hashes) == 0, "Leakage detected between train and val!"
    assert len(train_hashes & test_hashes) == 0, "Leakage detected between train and test!"
    assert len(val_hashes & test_hashes) == 0, "Leakage detected between val and test!"
    print("Zero leakage confirmed across train, val, and test!")

    print("[4/5] Copying files to destination directories...")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split_name, pairs in splits.items():
        s_img_dir = OUT_DIR / split_name / "images"
        s_lbl_dir = OUT_DIR / split_name / "labels"
        s_img_dir.mkdir(parents=True, exist_ok=True)
        s_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_p, lbl_p, _ in pairs:
            shutil.copy2(img_p, s_img_dir / img_p.name)
            if lbl_p.exists():
                shutil.copy2(lbl_p, s_lbl_dir / lbl_p.name)
            else:
                # empty file if background/no boxes
                (s_lbl_dir / f"{img_p.stem}.txt").touch()

    # Copy sample_video.mp4 if present in RAW_DIR to data/yolo/
    raw_video = RAW_DIR / "sample_video.mp4"
    if raw_video.exists():
        shutil.copy2(raw_video, OUT_DIR / "sample_video.mp4")
        print(f"Sample video copied to {OUT_DIR / 'sample_video.mp4'}")

    # Generate data.yaml
    data_yaml_content = f"""path: {OUT_DIR.as_posix()}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: pothole
"""
    (OUT_DIR / "data.yaml").write_text(data_yaml_content)
    print(f"data.yaml written to {OUT_DIR / 'data.yaml'}")

    print("[5/5] Generating visual verification plot with ground-truth boxes...")
    sample_indices = random.sample(range(len(splits["test"])), min(4, len(splits["test"])))
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for idx, s_idx in enumerate(sample_indices):
        img_p, lbl_p, n_b = splits["test"][s_idx]
        annotated_img = draw_yolo_boxes(img_p, lbl_p)
        axes[idx].imshow(annotated_img)
        axes[idx].set_title(f"{img_p.name} ({n_b} potholes)", fontsize=11, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle("Ground-Truth Pothole Annotations (Kaggle YOLO Dataset)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = PLOTS_DIR / "yolo_sample_annotations.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Verification plot saved to {plot_path}")

    # Save stats
    stats = {
        "dataset_source": "anggadwisunarto/potholes-detection-yolov8",
        "total_unique_images": len(unique_pairs),
        "duplicates_removed": duplicates_removed,
        "total_bounding_boxes": total_boxes,
        "avg_boxes_per_image": round(total_boxes / len(unique_pairs), 2),
        "splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "boxes_by_split": {
            "train": sum(p[2] for p in splits["train"]),
            "val": sum(p[2] for p in splits["val"]),
            "test": sum(p[2] for p in splits["test"]),
        },
    }
    with open(METRICS_DIR / "yolo_dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n================ DATASET VERIFICATION SUMMARY ================")
    print(f"  Total Unique Images:   {stats['total_unique_images']}")
    print(f"  Duplicates Removed:    {stats['duplicates_removed']}")
    print(f"  Total Labeled Boxes:   {stats['total_bounding_boxes']}")
    print(f"  Train Split (70%):     {stats['splits']['train']} images ({stats['boxes_by_split']['train']} boxes)")
    print(f"  Val Split (15%):       {stats['splits']['val']} images ({stats['boxes_by_split']['val']} boxes)")
    print(f"  Test Split (15%):      {stats['splits']['test']} images ({stats['boxes_by_split']['test']} boxes)")
    print("==============================================================")


if __name__ == "__main__":
    main()

