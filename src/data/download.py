"""
src/data/download.py
---------------------
Download the Kaggle Pothole Detection dataset.

Dataset: atulyakumar98/pothole-detection-dataset
  ~2,790 images in YOLO format (class images + txt labels)
  Licence: CC0 Public Domain

Usage:
    python src/data/download.py

Authentication:
  Option 1 (recommended): Set KAGGLE_USERNAME and KAGGLE_KEY env vars.
  Option 2: Place ~/.kaggle/kaggle.json with your credentials.
  Obtain your API token from https://www.kaggle.com/settings → API → Create New Token

If Kaggle credentials are not found, the script prints manual download
instructions and exits with code 1.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SLUG  = "atulyakumar98/pothole-detection-dataset"
EXPECTED_DIRS = ["train", "valid", "test"]   # typical YOLO dataset layout


def check_already_downloaded() -> bool:
    """Return True if the dataset looks already present."""
    # Check for any jpg files in raw/
    jpgs = list(RAW_DIR.rglob("*.jpg")) + list(RAW_DIR.rglob("*.jpeg")) + list(RAW_DIR.rglob("*.png"))
    return len(jpgs) >= 100


def download_kaggle() -> bool:
    """
    Try to download using the kaggle Python package.
    Returns True on success, False if credentials are missing.
    """
    try:
        import kaggle   # installed as part of requirements.txt
    except ImportError:
        print("[download] 'kaggle' package not installed. Run: pip install kaggle")
        return False

    # Kaggle API checks for credentials automatically in ~/.kaggle/kaggle.json
    # or KAGGLE_USERNAME / KAGGLE_KEY environment variables.
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_creds = (
        kaggle_json.exists()
        or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)
    )
    if not has_creds:
        return False

    print(f"[download] Downloading '{DATASET_SLUG}' from Kaggle …")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            DATASET_SLUG,
            path=str(RAW_DIR),
            unzip=True,
            quiet=False,
        )
        print(f"[download] ✓ Dataset downloaded to {RAW_DIR}")
        return True
    except Exception as e:
        print(f"[download] Kaggle download failed: {e}")
        return False


def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          MANUAL DOWNLOAD INSTRUCTIONS                           ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Go to:                                                       ║
║     https://www.kaggle.com/datasets/atulyakumar98/              ║
║                            pothole-detection-dataset             ║
║  2. Click "Download" (you need a free Kaggle account).          ║
║  3. Place the extracted folder contents into:                    ║
║     data/raw/                                                    ║
║     so that you have e.g. data/raw/train/images/ etc.           ║
║  4. Re-run this script to verify.                               ║
╚══════════════════════════════════════════════════════════════════╝
""")


def verify_dataset() -> dict:
    """
    Count images and labels after download.
    Returns a summary dict with real counts.
    """
    summary = {
        "total_images": 0,
        "total_labels": 0,
        "splits": {},
        "structure": "unknown",
    }

    # Count all image files recursively
    image_exts = {".jpg", ".jpeg", ".png"}
    all_images = [
        p for p in RAW_DIR.rglob("*")
        if p.suffix.lower() in image_exts
    ]
    all_labels = list(RAW_DIR.rglob("*.txt"))

    summary["total_images"] = len(all_images)
    summary["total_labels"] = len(all_labels)

    # Detect structure
    if (RAW_DIR / "train").exists():
        summary["structure"] = "yolo_split"
        for split in ["train", "valid", "test"]:
            split_dir = RAW_DIR / split
            if split_dir.exists():
                imgs = list((split_dir / "images").rglob("*")) if (split_dir / "images").exists() else list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.png"))
                imgs = [p for p in imgs if p.suffix.lower() in image_exts]
                summary["splits"][split] = len(imgs)
    elif (RAW_DIR / "pothole").exists() or (RAW_DIR / "normal").exists():
        summary["structure"] = "class_folders"
        for cls in ["pothole", "normal", "No Pothole", "Pothole"]:
            d = RAW_DIR / cls
            if d.exists():
                imgs = [p for p in d.rglob("*") if p.suffix.lower() in image_exts]
                summary["splits"][cls] = len(imgs)

    print("\n" + "="*55)
    print("  DATASET VERIFICATION")
    print("="*55)
    print(f"  Structure detected : {summary['structure']}")
    print(f"  Total images       : {summary['total_images']}")
    print(f"  Total label files  : {summary['total_labels']}")
    for k, v in summary["splits"].items():
        print(f"  Split/Class '{k}' : {v} images")
    print("="*55 + "\n")

    return summary


def main():
    if check_already_downloaded():
        print("[download] Dataset already present — skipping download.")
    else:
        success = download_kaggle()
        if not success:
            print_manual_instructions()
            sys.exit(1)

    summary = verify_dataset()

    if summary["total_images"] < 50:
        print("[download] ⚠ Very few images found. Check the data/raw/ directory.")
        sys.exit(1)

    print("[download] Dataset ready.")
    return summary


if __name__ == "__main__":
    main()

