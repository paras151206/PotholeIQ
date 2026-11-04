# PotholeIQ — Viva Guide

> This document explains every module in plain language, gives key equations,
> lists design choices with reasons, and provides 5 likely viva questions with
> strong answers.  Read this alongside the code, not instead of it.

---

## M0 — Repo Setup

### What was built
A reproducible project skeleton: directory structure, `config.yaml` for
all hyperparameters, `requirements.txt`, `.gitignore`, seed utilities, and
metric/plotting helpers.

### Key design choices
| Choice | Reason |
|--------|--------|
| `config.yaml` for all hyperparams | No hard-coded numbers in code; change one file, all scripts update |
| Fixed seed = 42 everywhere | Reproducibility — same seed → same train/val/test split, same weight initialisation |
| `worker_init_fn` in DataLoader | Each DataLoader worker needs its own seed or they produce correlated batches |
| `cuDNN deterministic = True` | Convolution algorithms have non-deterministic variants; this forces the reproducible path |

### Key equations
None specific to M0, but remember:
- **Cross-entropy loss** (used in classifiers): $\mathcal{L}_{CE} = -\sum_c y_c \log(\hat{p}_c)$
- **Binary cross-entropy** (used in segmentation): $\mathcal{L}_{BCE} = -[y \log \hat{p} + (1-y)\log(1-\hat{p})]$

### Viva Q&A

**Q1: Why do you need to set `torch.cuda.manual_seed_all` AND `torch.manual_seed`?**  
A: `torch.manual_seed` seeds the CPU RNG. CUDA operations use a *separate* GPU RNG;
`cuda.manual_seed_all` seeds it on every GPU. Missing either one leaves a source of
randomness unseeded.

**Q2: What does `cuDNN benchmark=False` do and what does it cost?**  
A: When `benchmark=True`, cuDNN runs several convolution algorithms at startup and
picks the fastest for your input size. This is not deterministic. Setting it to
`False` forces a fixed (reproducible) algorithm selection, at a typical cost of 5–15%
training speed.

**Q3: Why keep hyperparameters in a YAML file instead of argparse?**  
A: A YAML file is version-controlled — every git commit records the exact
hyperparameters used in that experiment. With argparse you'd have to remember to
log every flag; with YAML you get it for free.

**Q4: What is a `.gitkeep` file?**  
A: Git does not track empty directories. A `.gitkeep` (empty file) forces git to
include the directory in the repo so collaborators get the correct folder structure
when they clone.

**Q5: What is the purpose of `worker_init_fn` in a DataLoader?**  
A: PyTorch forks worker processes from the main process. All workers start with the
*same* NumPy/Python random state (the parent's state at fork time). Without
`worker_init_fn`, all workers generate identical random augmentations. The fix:
each worker derives a unique seed from its `worker_id`.

---

## M1 — Data (to be filled after M1 completes)

---

## M2 — Baseline (to be filled after M2 completes)

---

## M3 — Custom CNN (to be filled after M3 completes)

---

## M4 — Transfer Learning (to be filled after M4 completes)

---

## M5 — Segmentation U-Net (to be filled after M5 completes)

---

## M6 — Grad-CAM Explainability (to be filled after M6 completes)

---

## M7 — Siamese Network (to be filled after M7 completes)

---

## M8 — Autoencoder Anomaly Detection (to be filled after M8 completes)

---

## M9 — LSTM Trend (optional, to be filled if built)

---

## M10 — Streamlit App (to be filled after M10 completes)

