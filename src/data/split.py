"""
src/data/split.py
------------------
Stratified train/val/test split for the PotholeIQ dataset.

This script:
  1. Scans data/raw/ to discover all images and their labels.
  2. Creates a master DataFrame with real paths and real labels.
  3. Applies stratified split (70/15/15) with fixed seed=42.
  4. Verifies NO data leakage (zero filename overlap between splits).
  5. Saves train.csv, val.csv, test.csv to data/splits/.
  6. Also creates seg_train.csv, seg_val.csv, seg_test.csv that include
     bbox_path column pointing to YOLO .txt files.

Usage:
    python src/data/split.py

Output files (in data/splits/):
    train.csv, val.csv, test.csv       — columns: filepath, label
    seg_train.csv, seg_val.csv, seg_test.csv — columns: filepath, label, bbox_path
    split_summary.json                 — statistics for README data card
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import CFG
from src.utils.seed import set_all_seeds

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = CFG.data.split_ratios   # e.g. 0.70, 0.15, 0.15
SEED = CFG.seed
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ─────────────────────────── Dataset Discovery ───────────────────────────────

def discover_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    Auto-detect the dataset structure and return a DataFrame with
    columns [filepath, label, bbox_path].

    Supports three common layouts:
      A. YOLO detection split: raw/train/images/*.jpg + raw/train/labels/*.txt
      B. Class folders: raw/pothole/*.jpg + raw/normal/*.jpg (or Normal/Pothole)
      C. Flat with YOLO labels: raw/images/*.jpg + raw/labels/*.txt

    Label encoding: pothole=1, normal=0
    """
    records = []

    # ── Layout A: YOLO split folders ─────────────────────────────────────────
    yolo_splits = []
    for split_name in ["train", "valid", "test", "Train", "Valid", "Test"]:
        sd = raw_dir / split_name
        if sd.exists() and (sd / "images").exists():
            yolo_splits.append(sd)

    if yolo_splits:
        print(f"[split] Detected YOLO layout with {len(yolo_splits)} split folder(s)")
        for sd in yolo_splits:
            img_dir   = sd / "images"
            label_dir = sd / "labels"
            for img_path in img_dir.rglob("*"):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                # Determine class from YOLO label: if .txt exists and has content → pothole
                txt_path = label_dir / (img_path.stem + ".txt")
                if txt_path.exists():
                    content = txt_path.read_text().strip()
                    label = 1 if content else 0   # has annotation → pothole
                    bbox_path = str(txt_path) if content else ""
                else:
                    # Try to find in any labels subfolder
                    label = 0
                    bbox_path = ""
                records.append({
                    "filepath":  str(img_path),
                    "label":     label,
                    "bbox_path": bbox_path,
                })
        if records:
            return pd.DataFrame(records)

    # ── Layout B: Class folders ───────────────────────────────────────────────
    class_map = {}
    for cls_name in raw_dir.iterdir():
        if not cls_name.is_dir():
            continue
        name_lower = cls_name.name.lower().replace(" ", "_").replace("-", "_")
        if any(x in name_lower for x in ["pothole", "damage", "defect"]):
            class_map[cls_name] = 1
        elif any(x in name_lower for x in ["normal", "no_pothole", "plain", "good", "nodamage"]):
            class_map[cls_name] = 0

    if class_map:
        print(f"[split] Detected class-folder layout: {list(class_map.keys())}")
        for folder, label in class_map.items():
            for img_path in folder.rglob("*"):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                # Look for matching YOLO label
                labels_root = raw_dir / "labels"
                txt_path = labels_root / folder.name / (img_path.stem + ".txt")
                if not txt_path.exists():
                    txt_path = img_path.with_suffix(".txt")
                bbox_path = str(txt_path) if txt_path.exists() and txt_path.read_text().strip() else ""
                records.append({
                    "filepath":  str(img_path),
                    "label":     label,
                    "bbox_path": bbox_path,
                })
        if records:
            return pd.DataFrame(records)

    # ── Layout C: Flat images + labels/ ──────────────────────────────────────
    img_dir   = raw_dir / "images"
    label_dir = raw_dir / "labels"
    if not img_dir.exists():
        img_dir = raw_dir  # fallback: images directly in raw/

    if img_dir.exists():
        print(f"[split] Detected flat layout: {img_dir}")
        for img_path in img_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            txt_path = (label_dir / (img_path.stem + ".txt")) if label_dir.exists() else Path("")
            if txt_path.exists():
                content = txt_path.read_text().strip()
                label    = 1 if content else 0
                bbox_p   = str(txt_path) if content else ""
            else:
                label  = 0
                bbox_p = ""
            records.append({
                "filepath":  str(img_path),
                "label":     label,
                "bbox_path": bbox_p,
            })

    if not records:
        raise FileNotFoundError(
            f"No images found in {raw_dir}. "
            "Run src/data/download.py first."
        )

    return pd.DataFrame(records)


# ─────────────────────────── Split & Verify ──────────────────────────────────

def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split.

    Stratification ensures the class balance in each split mirrors the
    overall dataset balance.  Using a fixed seed makes the split
    reproducible — given the same input CSV, you always get the same split.
    """
    # First split: train vs (val + test)
    val_test_size = VAL_RATIO + TEST_RATIO
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_size,
        stratify=df["label"],
        random_state=SEED,
    )

    # Second split: val vs test  (val_ratio / (val+test) fraction)
    relative_val = VAL_RATIO / val_test_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - relative_val,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> bool:
    """
    Check that no image filename appears in more than one split.
    Uses stem (filename without extension) for robust matching.
    Returns True if clean, raises AssertionError if leakage detected.
    """
    def stems(df):
        return set(Path(p).stem for p in df["filepath"])

    train_s = stems(train_df)
    val_s   = stems(val_df)
    test_s  = stems(test_df)

    tv = train_s & val_s
    tt = train_s & test_s
    vt = val_s   & test_s

    if tv or tt or vt:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED!\n"
            f"  train∩val:  {len(tv)} files\n"
            f"  train∩test: {len(tt)} files\n"
            f"  val∩test:   {len(vt)} files"
        )
    print("[split] ✓ No data leakage detected between splits.")
    return True


def print_split_stats(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    full_df:  pd.DataFrame,
) -> dict:
    """Print a table of split sizes and class balance, return as dict."""
    summary = {}
    for name, df in [("full", full_df), ("train", train_df), ("val", val_df), ("test", test_df)]:
        n_total   = len(df)
        n_pothole = (df["label"] == 1).sum()
        n_normal  = (df["label"] == 0).sum()
        pct       = 100 * n_pothole / max(n_total, 1)
        summary[name] = {
            "total": int(n_total),
            "pothole": int(n_pothole),
            "normal": int(n_normal),
            "pothole_pct": round(pct, 1),
        }
        print(f"  {name:6s}: {n_total:5d} images | "
              f"pothole={n_pothole} ({pct:.1f}%) | normal={n_normal}")

    return summary


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    set_all_seeds(SEED)
    print(f"\n{'='*55}")
    print("  PotholeIQ — Stratified Split")
    print(f"{'='*55}")

    # 1. Discover dataset
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"[split] ERROR: {RAW_DIR} is empty. Run src/data/download.py first.")
        sys.exit(1)

    full_df = discover_dataset(RAW_DIR)
    print(f"\n[split] Found {len(full_df)} images total.")
    print(f"[split] Labels: pothole={( full_df['label']==1).sum()}, normal={(full_df['label']==0).sum()}")

    if len(full_df) < 50:
        print("[split] ⚠ Too few images. Check data/raw/ directory.")
        sys.exit(1)

    # 2. Stratified split
    train_df, val_df, test_df = stratified_split(full_df)

    # 3. Leakage check
    verify_no_leakage(train_df, val_df, test_df)

    # 4. Print stats
    print(f"\n{'─'*55}")
    print("  Split Statistics")
    print(f"{'─'*55}")
    summary = print_split_stats(train_df, val_df, test_df, full_df)

    # 5. Save CSV files
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # Classification CSV (filepath, label)
        cls_path = SPLITS_DIR / f"{name}.csv"
        df[["filepath", "label"]].to_csv(cls_path, index=False)
        print(f"[split] Saved → {cls_path}")

        # Segmentation CSV (filepath, label, bbox_path)
        seg_path = SPLITS_DIR / f"seg_{name}.csv"
        df.to_csv(seg_path, index=False)
        print(f"[split] Saved → {seg_path}")

    # 6. Save summary
    summary_path = SPLITS_DIR / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[split] Saved → {summary_path}")

    print(f"\n{'='*55}")
    print("  Split complete. Use these CSVs in all training scripts.")
    print(f"{'='*55}\n")

    return summary


if __name__ == "__main__":
    main()

