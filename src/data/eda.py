"""
src/data/eda.py
----------------
Exploratory Data Analysis for the PotholeIQ dataset.

Generates and saves the following plots to results/plots/:
  1. class_distribution.png  — bar chart of pothole vs normal counts per split
  2. sample_images.png        — 5×2 grid of sample images per class
  3. image_size_distribution.png — histogram of original image H and W
  4. pixel_intensity.png      — mean per-channel pixel intensity distribution
  5. bbox_area_distribution.png — distribution of bounding box areas (pothole images only)

Also prints key statistics for the README data card.

Usage:
    python src/data/eda.py

Requires that src/data/split.py has already been run (needs data/splits/*.csv).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on headless servers/Colab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.config import CFG

SPLITS_DIR = ROOT / Path(CFG.paths.splits_dir)
PLOTS_DIR  = ROOT / Path(CFG.paths.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["normal", "pothole"]
COLORS      = ["steelblue", "tomato"]


# ─────────────────────────── Helpers ─────────────────────────────────────────

def load_all_splits() -> dict[str, pd.DataFrame]:
    dfs = {}
    for name in ["train", "val", "test"]:
        path = SPLITS_DIR / f"{name}.csv"
        if path.exists():
            dfs[name] = pd.read_csv(path)
        else:
            print(f"[eda] ⚠ {path} not found. Run src/data/split.py first.")
    return dfs


# ─────────────────────────── Plot 1: Class Distribution ──────────────────────

def plot_class_distribution(dfs: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(1, len(dfs), figsize=(4 * len(dfs), 4), sharey=False)
    if len(dfs) == 1:
        axes = [axes]

    for ax, (split, df) in zip(axes, dfs.items()):
        counts = df["label"].value_counts().sort_index()
        bars = ax.bar(CLASS_NAMES, [counts.get(0, 0), counts.get(1, 0)], color=COLORS, edgecolor="black")
        ax.set_title(f"{split.capitalize()} split", fontsize=12)
        ax.set_ylabel("Count")
        ax.set_xlabel("Class")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 2, str(int(h)),
                    ha="center", va="bottom", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Class Distribution per Split", fontsize=14, y=1.02)
    fig.tight_layout()
    out = PLOTS_DIR / "class_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved → {out}")


# ─────────────────────────── Plot 2: Sample Images ───────────────────────────

def plot_sample_images(dfs: dict[str, pd.DataFrame], n_per_class: int = 5):
    """Show n_per_class examples from each class (using the train split)."""
    df = dfs.get("train", next(iter(dfs.values())))

    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 7))
    fig.suptitle("Sample Images — Train Split", fontsize=14)

    for row, label in enumerate([0, 1]):
        cls_df = df[df["label"] == label].sample(
            n=min(n_per_class, len(df[df["label"] == label])),
            random_state=42,
        )
        for col, (_, row_data) in enumerate(cls_df.iterrows()):
            ax = axes[row][col]
            fp = Path(row_data["filepath"])
            if not fp.is_absolute():
                fp = ROOT / fp
            try:
                img = Image.open(fp).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("lightgray")
                ax.text(0.5, 0.5, "err", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(CLASS_NAMES[label], color=COLORS[label], fontsize=8)
            ax.axis("off")

    fig.tight_layout()
    out = PLOTS_DIR / "sample_images.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved → {out}")


# ─────────────────────────── Plot 3: Image Sizes ─────────────────────────────

def plot_image_sizes(dfs: dict[str, pd.DataFrame], max_samples: int = 500):
    df = dfs.get("train", next(iter(dfs.values())))
    sample_df = df.sample(n=min(max_samples, len(df)), random_state=42)

    widths, heights = [], []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="[eda] Reading sizes"):
        fp = Path(row["filepath"])
        if not fp.is_absolute():
            fp = ROOT / fp
        try:
            with Image.open(fp) as img:
                w, h = img.size
            widths.append(w)
            heights.append(h)
        except Exception:
            pass

    if not widths:
        print("[eda] ⚠ Could not read any images for size distribution.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(widths,  bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.set(title="Width Distribution", xlabel="Width (px)", ylabel="Count")
    ax1.axvline(np.mean(widths), color="red", linestyle="--", label=f"Mean={np.mean(widths):.0f}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.hist(heights, bins=30, color="tomato",    edgecolor="black", alpha=0.8)
    ax2.set(title="Height Distribution", xlabel="Height (px)", ylabel="Count")
    ax2.axvline(np.mean(heights), color="blue", linestyle="--", label=f"Mean={np.mean(heights):.0f}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Original Image Size Distribution (n={len(widths)} sampled)", fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / "image_size_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved → {out}")
    print(f"[eda] Size stats: W={np.mean(widths):.0f}±{np.std(widths):.0f}, "
          f"H={np.mean(heights):.0f}±{np.std(heights):.0f}")


# ─────────────────────────── Plot 4: Pixel Intensity ─────────────────────────

def plot_pixel_intensity(dfs: dict[str, pd.DataFrame], max_samples: int = 200):
    df = dfs.get("train", next(iter(dfs.values())))
    sample_df = df.sample(n=min(max_samples, len(df)), random_state=42)

    means = {cls: {c: [] for c in ["R", "G", "B"]} for cls in [0, 1]}

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="[eda] Computing pixel stats"):
        fp = Path(row["filepath"])
        if not fp.is_absolute():
            fp = ROOT / fp
        try:
            arr = np.array(Image.open(fp).convert("RGB").resize((64, 64))) / 255.0
            lbl = row["label"]
            for ci, ch in enumerate(["R", "G", "B"]):
                means[lbl][ch].append(arr[:, :, ci].mean())
        except Exception:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ci, ch in enumerate(["R", "G", "B"]):
        ax = axes[ci]
        for lbl, color in [(0, "steelblue"), (1, "tomato")]:
            vals = means[lbl][ch]
            if vals:
                ax.hist(vals, bins=25, alpha=0.6, color=color,
                        label=CLASS_NAMES[lbl], edgecolor="black")
        ax.set(title=f"{ch} channel mean intensity", xlabel="Mean pixel value", ylabel="Count")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Per-channel Pixel Intensity (normal vs pothole)", fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / "pixel_intensity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved → {out}")


# ─────────────────────────── Plot 5: BBox Area Distribution ──────────────────

def plot_bbox_area(dfs: dict[str, pd.DataFrame]):
    """
    For images with YOLO bounding boxes, plot the distribution of pothole
    area ratio (bbox_area / image_area).  Also print severity bucket counts.
    """
    # Look for seg splits (they have bbox_path column)
    seg_path = SPLITS_DIR / "seg_train.csv"
    if not seg_path.exists():
        print("[eda] seg_train.csv not found — skipping bbox area plot.")
        return

    df = pd.read_csv(seg_path)
    if "bbox_path" not in df.columns:
        return

    pothole_df = df[(df["label"] == 1) & (df["bbox_path"].notna()) & (df["bbox_path"] != "")]
    area_ratios = []

    for _, row in tqdm(pothole_df.sample(min(500, len(pothole_df)), random_state=42).iterrows(),
                       desc="[eda] Computing bbox areas"):
        try:
            bp = Path(row["bbox_path"])
            if not bp.is_absolute():
                bp = ROOT / bp
            if not bp.exists():
                continue
            fp = Path(row["filepath"])
            if not fp.is_absolute():
                fp = ROOT / fp
            with Image.open(fp) as img:
                img_w, img_h = img.size
            with open(bp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bw, bh = float(parts[3]), float(parts[4])
                        area_ratios.append(bw * bh)
        except Exception:
            pass

    if not area_ratios:
        print("[eda] No bbox areas computed — skipping bbox plot.")
        return

    thresholds = CFG.unet.severity_thresholds
    low_max = thresholds.low_max
    med_max = thresholds.medium_max

    n_low  = sum(1 for r in area_ratios if r < low_max)
    n_med  = sum(1 for r in area_ratios if low_max <= r < med_max)
    n_high = sum(1 for r in area_ratios if r >= med_max)
    total  = len(area_ratios)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(area_ratios, bins=40, color="tomato", edgecolor="black", alpha=0.8)
    ax.axvline(low_max, color="orange", linestyle="--", label=f"Low/Med boundary ({low_max:.0%})")
    ax.axvline(med_max, color="red",    linestyle="--", label=f"Med/High boundary ({med_max:.0%})")
    ax.set(title="Pothole Bounding Box Area Ratio Distribution",
           xlabel="bbox_area / image_area", ylabel="Count")
    ax.legend()
    ax.grid(alpha=0.3)

    textstr = (f"Low (<{low_max:.0%}):    {n_low} ({100*n_low/total:.1f}%)\n"
               f"Medium ({low_max:.0%}–{med_max:.0%}): {n_med} ({100*n_med/total:.1f}%)\n"
               f"High (>{med_max:.0%}):  {n_high} ({100*n_high/total:.1f}%)")
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, va="top", ha="right",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    out = PLOTS_DIR / "bbox_area_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved → {out}")
    print(f"[eda] Severity distribution: Low={n_low}, Medium={n_med}, High={n_high}")


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    print(f"\n{'='*55}")
    print("  PotholeIQ — EDA")
    print(f"{'='*55}\n")

    dfs = load_all_splits()
    if not dfs:
        print("[eda] No splits found. Run src/data/split.py first.")
        sys.exit(1)

    # Print summary to console (for README data card)
    print("\n── Split Summary ──")
    for split, df in dfs.items():
        n = len(df)
        p = (df["label"] == 1).sum()
        no = (df["label"] == 0).sum()
        print(f"  {split:5s}: {n:5d} images  pothole={p} ({100*p/n:.1f}%)  normal={no} ({100*no/n:.1f}%)")

    print("\n── Generating plots ──")
    plot_class_distribution(dfs)
    plot_sample_images(dfs)
    plot_image_sizes(dfs)
    plot_pixel_intensity(dfs)
    plot_bbox_area(dfs)

    print(f"\n[eda] All plots saved to {PLOTS_DIR}")
    print("[eda] ✓ EDA complete.\n")


if __name__ == "__main__":
    main()

