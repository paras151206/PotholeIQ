"""
src/data/dataset.py
--------------------
PyTorch Dataset classes for PotholeIQ.

Three dataset classes:
  PotholeDataset     — binary classification (pothole / normal)
  PotholeSegDataset  — segmentation; generates mask from YOLO bbox if no mask file
  SiameseDataset     — positive = same image with two augmentations;
                       negative = different-class image pair

Shape contracts (all models use image_size from config):
  image tensor : (3, H, W)  float32  normalised to ImageNet stats
  seg mask     : (1, H, W)  float32  binary {0.0, 1.0}
  siamese pair : two tensors of shape (3, H, W)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# -- Project utilities ---------------------------------------------------------
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.config import CFG

H, W = CFG.data.image_size          # e.g. [128, 128]
MEAN = CFG.augmentation.normalize_mean
STD  = CFG.augmentation.normalize_std

# --------------------------- Transforms --------------------------------------

def get_transforms(split: str) -> T.Compose:
    """
    Return the image transform pipeline for a given split.

    Train split applies data augmentation (flips, rotation, colour jitter,
    random erasing) to regularise the model.  Val/test use deterministic
    resize + normalise only so evaluation is reproducible.

    Parameters
    ----------
    split : "train" | "val" | "test"
    """
    aug = CFG.augmentation

    if split == "train":
        ops = [
            T.Resize((H, W)),
            T.RandomHorizontalFlip(p=0.5 if aug.horizontal_flip else 0.0),
            T.RandomVerticalFlip(p=0.5 if aug.vertical_flip else 0.0),
            T.RandomRotation(degrees=aug.rotation_degrees),
            T.ColorJitter(
                brightness=aug.color_jitter.brightness,
                contrast=aug.color_jitter.contrast,
                saturation=aug.color_jitter.saturation,
                hue=aug.color_jitter.hue,
            ),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
        if aug.random_erasing:
            ops.append(T.RandomErasing(p=0.3, scale=(0.02, 0.1)))
        return T.Compose(ops)

    else:  # val / test — deterministic
        return T.Compose([
            T.Resize((H, W)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])


def get_aug_pair_transform() -> T.Compose:
    """
    Strong augmentation transform that returns a heavily-augmented view.
    Used by SiameseDataset to generate positive pairs.
    Two calls on the same image -> two different augmented views.
    """
    aug = CFG.augmentation
    return T.Compose([
        T.Resize((H, W)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


# --------------------------- Classification Dataset --------------------------

class PotholeDataset(Dataset):
    """
    Binary classification dataset loaded from a CSV split file.

    CSV format (created by src/data/split.py):
        filepath  : path to image (relative to repo root or absolute)
        label     : 0 = normal road, 1 = pothole

    Parameters
    ----------
    csv_path  : path to split CSV
    transform : torchvision transform applied to each image
    root      : repo root (default: auto-detected from this file's location)
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: T.Compose | None = None,
        root: str | Path | None = None,
    ):
        self.root = Path(root) if root else ROOT
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Validate required columns
        for col in ("filepath", "label"):
            if col not in self.df.columns:
                raise ValueError(f"CSV missing required column: '{col}'")

        self.df["label"] = self.df["label"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        fp  = Path(row["filepath"])
        # Support both absolute and repo-relative paths
        if not fp.is_absolute():
            fp = self.root / fp

        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, int(row["label"])

    @property
    def labels(self) -> list[int]:
        """Return all labels as a Python list (used for class-weight computation)."""
        return self.df["label"].tolist()


# --------------------------- Segmentation Dataset ----------------------------

def _yolo_to_mask(
    bbox_line: str,
    img_w: int,
    img_h: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """
    Convert a single YOLO-format annotation line to a binary mask of size
    (out_h, out_w).  Fills a white rectangle at the bbox location.

    YOLO format: class x_center y_center width height  (all 0–1 normalised)

    If the YOLO line is empty or malformed, returns a zero mask (no pothole).
    This is a known limitation — real pixel masks would give better IoU.
    """
    mask = np.zeros((out_h, out_w), dtype=np.float32)
    parts = bbox_line.strip().split()
    if len(parts) < 5:
        return mask

    _, xc, yc, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    # Convert to pixel coords in output resolution
    x1 = int((xc - bw / 2) * out_w)
    y1 = int((yc - bh / 2) * out_h)
    x2 = int((xc + bw / 2) * out_w)
    y2 = int((yc + bh / 2) * out_h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(out_w, x2), min(out_h, y2)
    mask[y1:y2, x1:x2] = 1.0
    return mask


class PotholeSegDataset(Dataset):
    """
    Segmentation dataset.  Returns (image_tensor, mask_tensor) pairs.

    CSV format:
        filepath   : path to image
        label      : 0 or 1  (1 = pothole, used to skip normal images)
        bbox_path  : path to YOLO .txt file (may be empty string for normal images)

    If bbox_path is provided, the YOLO annotations are rasterised to a binary
    mask (rectangular approximation — documented limitation in README).
    If bbox_path is empty, the mask is all zeros.

    mask_tensor shape: (1, H, W)  float32  binary {0, 1}
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_transform: T.Compose | None = None,
        root: str | Path | None = None,
    ):
        self.root = Path(root) if root else ROOT
        self.df = pd.read_csv(csv_path)
        self.img_transform = img_transform
        self.mask_resize = T.Resize((H, W), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        fp  = Path(row["filepath"])
        if not fp.is_absolute():
            fp = self.root / fp

        img = Image.open(fp).convert("RGB")
        orig_w, orig_h = img.size

        # Build mask
        mask = np.zeros((H, W), dtype=np.float32)
        bbox_path = str(row.get("bbox_path", "")).strip()
        if bbox_path and Path(bbox_path).exists():
            with open(bbox_path) as f:
                for line in f:
                    if line.strip():
                        m = _yolo_to_mask(line, orig_w, orig_h, H, W)
                        mask = np.maximum(mask, m)   # union of all bbox masks

        if self.img_transform:
            img = self.img_transform(img)
        else:
            img = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(MEAN, STD)])(img)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return img, mask_tensor


# --------------------------- Siamese Dataset ---------------------------------

class SiameseDataset(Dataset):
    """
    Siamese dataset for contrastive / metric learning.

    Positive pair (label=0, same):
        Take the same pothole image and apply two *different* random augmentations.
        This simulates the same pothole photographed under slightly different
        conditions.  ⚠️ Limitation: we do NOT have real same-pothole-over-time
        pairs; this is a synthetic surrogate documented in the README.

    Negative pair (label=1, different):
        A pothole image paired with a normal-road image.

    Returns: (img1, img2, is_different_int)
        is_different = 0 -> same class (positive pair)
        is_different = 1 -> different class (negative pair)
    This is the convention used in contrastive loss.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: T.Compose | None = None,
        root: str | Path | None = None,
        pos_neg_ratio: float = 0.5,  # fraction of samples that are positive pairs
    ):
        self.root = Path(root) if root else ROOT
        self.df   = pd.read_csv(csv_path)
        self.aug_transform = get_aug_pair_transform()
        self.val_transform = transform if transform else get_transforms("val")
        self.pos_neg_ratio = pos_neg_ratio

        self.pothole_idx = self.df.index[self.df["label"] == 1].tolist()
        self.normal_idx  = self.df.index[self.df["label"] == 0].tolist()

        if not self.pothole_idx or not self.normal_idx:
            raise ValueError("SiameseDataset requires BOTH classes to be present in the CSV.")

    def _load_img(self, idx: int) -> Image.Image:
        fp = Path(self.df.iloc[idx]["filepath"])
        if not fp.is_absolute():
            fp = self.root / fp
        return Image.open(fp).convert("RGB")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        is_positive = (np.random.rand() < self.pos_neg_ratio)

        if is_positive:
            # Positive pair: same pothole image, two different augmentations
            anchor_idx = self.pothole_idx[idx % len(self.pothole_idx)]
            img = self._load_img(anchor_idx)
            img1 = self.aug_transform(img)
            img2 = self.aug_transform(img)
            label = 0  # same (not different)
        else:
            # Negative pair: pothole + normal
            p_idx = self.pothole_idx[idx % len(self.pothole_idx)]
            n_idx = self.normal_idx[np.random.randint(len(self.normal_idx))]
            img1 = self.aug_transform(self._load_img(p_idx))
            img2 = self.val_transform(self._load_img(n_idx))
            label = 1  # different

        return img1, img2, label


# --------------------------- Autoencoder Dataset -----------------------------

class NormalRoadDataset(Dataset):
    """
    Dataset containing ONLY normal (non-pothole) road images.
    Used to train the autoencoder on the normal class only.
    At inference, potholes -> high reconstruction error -> anomaly.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: T.Compose | None = None,
        root: str | Path | None = None,
    ):
        self.root = Path(root) if root else ROOT
        df_all   = pd.read_csv(csv_path)
        self.df  = df_all[df_all["label"] == 0].reset_index(drop=True)
        self.transform = transform

        if len(self.df) == 0:
            raise ValueError("NormalRoadDataset: no normal (label=0) images found.")

        print(f"[NormalRoadDataset] Loaded {len(self.df)} normal images from {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fp = Path(self.df.iloc[idx]["filepath"])
        if not fp.is_absolute():
            fp = self.root / fp
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img  # no label; AE is unsupervised

