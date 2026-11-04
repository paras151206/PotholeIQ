"""
src/utils/seed.py
-----------------
Centralised random-seed utility.

Why this matters (viva question): Deep learning has many sources of
randomness — Python's random module, NumPy, PyTorch CPU ops, PyTorch
CUDA ops, and even DataLoader worker initialisation.  Fixing all of
them makes every training run bit-for-bit reproducible on the same
hardware.

Usage:
    from src.utils.seed import set_all_seeds
    set_all_seeds(42)      # call this BEFORE anything else in a script
"""

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """
    Fix all sources of randomness used in this project.

    Parameters
    ----------
    seed : int
        The global seed value.  Stored in config.yaml under ``seed``.
    """
    # 1. Python built-in random (used by e.g. random.shuffle)
    random.seed(seed)

    # 2. NumPy (used by scikit-learn, pandas sampling, albumentations)
    np.random.seed(seed)

    # 3. PyTorch CPU operations
    torch.manual_seed(seed)

    # 4. PyTorch CUDA operations (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # multi-GPU

    # 5. cuDNN deterministic mode.
    #    Trade-off: deterministic=True can be ~10 % slower because cuDNN
    #    picks the reproducible (not always fastest) convolution algorithm.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. Environment variable read by some C extensions
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[seed] All random seeds fixed to {seed}.")


def worker_init_fn(worker_id: int) -> None:
    """
    Pass to DataLoader(worker_init_fn=...) so each worker gets a
    different-but-deterministic seed derived from the global seed.
    Without this, workers share the same NumPy seed → correlated batches.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
