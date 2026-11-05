# PotholeIQ — Oral Viva Defense Guide

> Every module explained in plain language, key mathematical equations, design choices, negative results analysis, and 5 viva Q&As per module.
> Authored by Paras Panchal for B.Tech CSE (AI/ML) Deep Learning course project defense.

---

## M0 — Repo Setup & Engineering Foundations

### What was built
Config-driven skeleton: `config.yaml`, `requirements.txt`, seed utilities (`set_all_seeds`, `worker_init_fn`), metric helpers, plotting helpers, and `smoke_test.py`.

### Key design choices
| Choice | Reason |
|---|---|
| `config.yaml` for all hyperparams | Version-controlled experiments; no magic numbers in code |
| `seed=42` everywhere | Reproducibility — identical split, identical weight init, deterministic augmentations |
| `worker_init_fn` in DataLoader | Forked workers share RNG state; without this, all workers produce identical augmentations |
| `cuDNN deterministic=True` | Forces reproducible (not benchmarked) convolution algorithm |

### Viva Q&A

**Q1: Why do you need `worker_init_fn` in a DataLoader?**
> When PyTorch forks workers, all inherit the same NumPy/Python RNG state from the parent process. Without `worker_init_fn`, every worker applies *identical* random augmentations each epoch, defeating the purpose of multi-worker augmentation. The fix derives a unique seed from each worker's ID (`base_seed + worker_id`).

**Q2: What is the cost of `torch.backends.cudnn.deterministic=True`?**
> cuDNN normally benchmarks several convolution algorithms at startup and selects the fastest. `deterministic=True` forces a fixed, mathematically reproducible algorithm, typically running 5–15% slower. It is mandatory for scientific reproducibility.

**Q3: Why keep hyperparameters in YAML vs argparse?**
> YAML files are checked into git — every git commit permanently records the exact hyperparameters used. With argparse you must remember to log every flag manually. With YAML you get a full experiment audit trail for free.

**Q4: What is the purpose of `smoke_test.py`?**
> It runs rapid synthetic forward passes through all 12 model architectures, loss functions, and detectors without requiring full datasets. If any layer shape, tensor dimension, or dependency is broken, it fails immediately in under 15 seconds.

**Q5: Explain the difference between `torch.manual_seed` and `torch.cuda.manual_seed_all`.**
> PyTorch maintains separate RNG states for CPU and each GPU device. `manual_seed` seeds only the CPU RNG. `cuda.manual_seed_all` seeds the CUDA RNG on every attached GPU device. Both must be set for full cross-device reproducibility.

---

## M1 — Data Pipeline & Leakage Prevention

### What was built
`dataset.py`, `download.py` (Kaggle API), `split.py` (stratified split + leakage check), `prepare_yolo_dataset.py` (MD5 deduplication + YOLO format verification).

### Datasets Summary
| Dataset | Source | Total Images | Annotations | Partitioning |
|---|---|:---:|---|---|
| **YOLO Detection** | `anggadwisunarto/potholes-detection-yolov8` | 1,590 unique (387 duplicates removed) | 5,684 YOLO `.txt` bounding boxes | 70% Train (1113) / 15% Val (238) / 15% Test (239) |
| **Classification** | `atulyakumar98/pothole-detection-dataset` | 681 | Binary labels (329 pothole / 352 normal) | 70% Train (476) / 15% Val (102) / 15% Test (103) |

### Key equations
**Data split sizes** ($N$ = total images, $r_1=0.70, r_2=0.15, r_3=0.15$):
$$N_{\text{train}} = \lfloor r_1 \cdot N \rfloor, \quad N_{\text{val}} = \lfloor r_2 \cdot N \rfloor, \quad N_{\text{test}} = N - N_{\text{train}} - N_{\text{val}}$$

**Normalized YOLO to pixel coordinates:**
$$x_1 = (x_c - w/2) \cdot W_{\text{img}}, \quad y_1 = (y_c - h/2) \cdot H_{\text{img}}$$
$$x_2 = (x_c + w/2) \cdot W_{\text{img}}, \quad y_2 = (y_c + h/2) \cdot H_{\text{img}}$$

### Viva Q&A

**Q1: Why use stratified splitting instead of random splitting?**
> Random splitting can produce imbalanced subsets by chance. Stratified splitting guarantees that the positive/negative ratio in train, val, and test partitions exactly matches the overall population distribution (~48.3% potholes).

**Q2: How did you ensure zero data leakage?**
> In both datasets, we computed cryptographic MD5 file hashes across all image files. In the YOLO dataset, 387 duplicate images were identified and removed. We then ran automated disjointness assertions ($\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$) to ensure zero overlap.

**Q3: What transforms do you apply during training vs validation?**
> Train: Resize $\to$ RandomHorizontalFlip $\to$ RandomRotation $\to$ ColorJitter $\to$ ToTensor $\to$ Normalize $\to$ RandomErasing. Validation/Test: Resize $\to$ ToTensor $\to$ Normalize only. Applying random augmentations at test time would introduce evaluation stochasticity.

**Q4: Why use ImageNet normalization statistics for road images?**
> Pretrained models like MobileNetV2 and YOLO backbones were pretrained with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Matching these statistics ensures input pixel distributions align with the filters learned during pretraining.

**Q5: Why did you remove 387 duplicate images from the YOLO dataset?**
> Web-scraped and municipal datasets often contain duplicate or burst-shot identical frames. If an image appears in both train and test, the model memorizes it and inflates test evaluation metrics (data leakage). Deduplication ensures legitimate test generalization.

---

## M2 — Baseline Multi-Layer Perceptron (TinyMLP)

### What was built
`baseline.py` — 2-hidden-layer MLP on flattened raw pixels ($3 \times 128 \times 128 = 49,152$ inputs); `train_baseline.py`.

### Architecture & Parameters
```
Input: (B, 3, 128, 128) → Flatten → (B, 49152)
Linear(49152, 512) → ReLU → Dropout(0.3) → (B, 512)
Linear(512, 128)   → ReLU → Dropout(0.3) → (B, 128)
Linear(128, 2)     → (B, 2) [logits]
Total Parameters: 25,232,258 (100.9 MB)
```

### Performance on 103 Test Images
- **Accuracy:** 72.82%
- **Macro F1:** 0.7250
- **ROC-AUC:** 0.8502

### Viva Q&A

**Q1: Why build an MLP baseline if you know CNNs are better?**
> In the universal machine learning workflow, you must establish an empirical lower bound. The MLP baseline tests whether road images are linearly or non-spatially separable and quantifies the exact gain provided by convolutional inductive biases.

**Q2: Why is flattening an image parameter-inefficient?**
> A single dense connection from $49,152$ inputs to $512$ neurons requires $49,152 \times 512 + 512 = 25,166,336$ weights in the first layer alone. It treats adjacent pixels as completely independent, discarding 2D spatial locality.

**Q3: What inductive bias does an MLP lack compared to a CNN?**
> MLPs lack **translation equivariance** (if a pothole shifts 5 pixels, an MLP sees entirely new input features) and **spatial locality** (pixels close together are correlated; distant pixels are not).

**Q4: How does weighted cross-entropy address class imbalance?**
> By weighting loss inversely proportional to class frequency: $w_c = N / (C \cdot N_c)$. Errors on the minority class produce proportionally larger gradients, preventing the network from defaulting to majority-class predictions.

**Q5: Why did the MLP achieve 72.82% accuracy despite having 25 million parameters?**
> Massive overparameterization without spatial inductive bias caused the MLP to memorize background road color and brightness rather than structural cavity contours.

---

## M3 — Custom 4-Stage CNN & Ablation Studies

### What was built
`cnn.py` — 4-stage convolutional network with BatchNorm, Dropout, and Global Average Pooling; `train_cnn.py`.

### Architecture Progression
```
Stage 1: Conv2d(3→32, k=3, p=1)   → BN → ReLU → MaxPool(2) → (B, 32, 64, 64)
Stage 2: Conv2d(32→64, k=3, p=1)  → BN → ReLU → MaxPool(2) → (B, 64, 32, 32)
Stage 3: Conv2d(64→128, k=3, p=1) → BN → ReLU → MaxPool(2) → (B, 128, 16, 16)
Stage 4: Conv2d(128→256, k=3, p=1)→ BN → ReLU → MaxPool(2) → (B, 256, 8, 8)
Head:    GlobalAvgPool2d → Linear(256→128) → ReLU → Dropout(0.4) → Linear(128→2)
Total Parameters: 425,026 (1.7 MB) — 98.3% parameter reduction vs MLP!
```

### Ablation Study Results (103 Test Images)
| Configuration | Test Accuracy | Macro F1 | ROC-AUC | Impact |
|---|:---:|:---:|:---:|---|
| **Full Custom CNN** | **92.23%** | **0.9223** | **0.9868** | **Best generalization** |
| No Augmentation | 86.41% | 0.8637 | 0.9438 | -5.86% F1 drop; severe overfitting |
| No BatchNorm | 88.35% | 0.8833 | 0.9453 | -3.90% F1 drop; slower convergence |
| No Dropout | 92.23% | 0.9223 | 0.9872 | Wider train-val loss gap |

### Viva Q&A

**Q1: What is Global Average Pooling (GAP) and why is it superior to flattening?**
> GAP computes the mean of each feature map: $(B, C, H, W) \to (B, C)$. Flattening yields $256 \times 8 \times 8 = 16,384$ features requiring millions of dense weights. GAP reduces features to 256, eliminates 95% of classifier parameters, acts as a structural regularizer, and makes the model spatially invariant.

**Q2: Why does Batch Normalization accelerate training?**
> BN normalizes each layer's activations across the batch to zero mean and unit variance. This reduces internal covariate shift, smooths the optimization landscape, enables higher learning rates, and provides a mild regularizing effect.

**Q3: Explain the overfit sanity check and why it is critical.**
> We train the network on a tiny 50-image subset for 20 epochs without regularization. The model should rapidly reach ~100% training accuracy. If it cannot, the architecture is fundamentally broken (e.g., gradient vanishing, wrong loss). Passing the check proves the model has sufficient representational capacity.

**Q4: What is CosineAnnealingLR and why use it over StepLR?**
> Cosine annealing smoothly decays the learning rate following a half-cosine curve: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T_{\max}))$. Unlike StepLR, which creates sudden loss spikes, cosine annealing allows high exploration early and smooth convergence late.

**Q5: What are the statistical limits of the ablation conclusions?**
> With a 103-image test split, small metric differences (such as Full CNN vs No Dropout at 92.23%) fall within the bootstrap margin of error. While data augmentation and BatchNorm show clear empirical improvements, fine-grained ablation differences must be interpreted with scientific modesty.

---

## M4 — Transfer Learning (MobileNetV2)

### What was built
`transfer.py` — MobileNetV2 with custom head; `train_transfer.py` — two-phase fine-tuning.

### Two-Phase Fine-Tuning Strategy
| Phase | Frozen Layers | LR | Epochs | Val F1 | Purpose |
|---|---|:---:|:---:|:---:|---|
| **Phase 1** | Backbone entirely frozen | $1 \times 10^{-3}$ | 5 | 0.9705 | Train new head while protecting pretrained ImageNet features |
| **Phase 2** | Last 3 InvertedResidual blocks unfrozen | $5 \times 10^{-5}$ | 6 | **0.9804** | Fine-tune high-level features for road surface damage |

### Performance on 103 Test Images
- **Accuracy:** 98.06%
- **Macro F1:** 0.9806
- **ROC-AUC:** 0.9985
- **95% Bootstrap F1 CI:** `[95.02%, 100.0%]`

### Viva Q&A

**Q1: What is transfer learning and why does it work for asphalt roads?**
> Transfer learning adapts a model pretrained on a massive corpus (ImageNet, 1.28M images) to a downstream domain. It works because low-level convolutional filters (edges, textures, gradients, color transitions) are universal visual primitives that transfer directly to road surface cracks.

**Q2: What is catastrophic forgetting and how do you prevent it?**
> Catastrophic forgetting occurs when large gradient updates overwrite pretrained feature weights, erasing generic visual representations. We prevent it by: (1) completely freezing the backbone during Phase 1, and (2) using a learning rate 20× smaller ($5 \times 10^{-5}$) in Phase 2.

**Q3: Which layers do you unfreeze in Phase 2 and why?**
> We unfreeze only the last 3 InvertedResidual blocks (blocks 16–18). Early layers capture generic edges and textures; later layers capture high-level semantic shapes. Unfreezing only the final blocks allows the network to adapt to road distress while keeping generic early filters intact.

**Q4: How does MobileNetV2 achieve efficiency?**
> Through **inverted residual bottlenecks** (narrow $\to$ wide $\to$ narrow) and **depthwise separable convolutions** (splitting standard convolution into spatial depthwise filtering and $1\times 1$ pointwise channel projection). This reduces FLOPs by 8–9× compared to standard convolutions with minimal accuracy loss.

**Q5: What does the 95% bootstrap confidence interval tell you about MobileNetV2 vs Custom CNN?**
> MobileNetV2 achieves a mean test F1 of 98.06% with CI `[95.02%, 100.0%]`, while Custom CNN achieves 92.23% with CI `[86.77%, 97.08%]`. While MobileNetV2 shows superior point metrics, their confidence intervals slightly overlap, meaning we claim strong empirical superiority rather than absolute statistical proof on a 103-image test split.

---

## YOLOv8n — Real-Time Object Detection (Images & Video)

### What was built
`prepare_yolo_dataset.py`, `train_yolo.py`, `notebooks/train_yolo_colab.ipynb`; fine-tuned YOLOv8n on 1,590 deduplicated images (5,684 bounding boxes).

### Test Set Performance (239 Held-Out Images, 887 Instances)
- **Precision (P):** 71.46%
- **Recall (R):** 55.34%
- **mAP @ 0.50:** 61.49%
- **mAP @ 0.50:0.95:** 36.18%
- **Inference Latency:** 18.0 ms / frame (~55 FPS on CPU)
- **Checkpoint Footprint:** 6.47 MB (`best.pt`)

### Key equations
**CIoU Loss (Complete Intersection over Union):**
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v$$
where $\rho$ is Euclidean distance between box centers, $c$ is diagonal length of the smallest enclosing box, and $v$ measures aspect ratio consistency:
$$v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$

**Mean Average Precision (mAP):**
$$\text{AP} = \int_0^1 p(r) \, dr, \quad \text{mAP} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c$$

### Viva Q&A

**Q1: Why did you transition from U-Net to YOLOv8n?**
> The dataset provided bounding-box coordinates, not pixel-level polygon masks. Training a U-Net on rasterized rectangular masks caused it to memorize rectangular bounding priors rather than true pothole morphology (Dice was artificially inflated to 0.9051). YOLOv8n is mathematically formulated for bounding-box supervision and predicts multi-instance locations, counts, and confidence scores.

**Q2: What is the difference between anchor-based and anchor-free detection?**
> Traditional detectors (YOLOv3, Faster R-CNN) predefine anchor boxes of fixed aspect ratios and predict offsets. YOLOv8 is anchor-free: it directly predicts the distance from the grid cell center to the 4 box boundaries ($l, r, t, b$) using Distribution Focal Loss. This eliminates manual anchor tuning and handles irregular pothole shapes better.

**Q3: Explain mAP@0.5 vs mAP@0.5:0.95.**
> mAP@0.5 measures Average Precision at a 50% IoU overlap threshold between prediction and ground truth (PASCAL VOC benchmark). mAP@0.5:0.95 averages AP across 10 IoU thresholds from 0.50 to 0.95 in 0.05 increments (COCO benchmark). It heavily penalizes sloppy box boundaries. Our model achieves 61.49% mAP@0.5 and 36.18% mAP@0.5:0.95.

**Q4: How does Non-Maximum Suppression (NMS) work during inference?**
> Multiple grid cells can detect the same pothole. NMS sorts all candidate boxes by confidence, selects the highest-scoring box, and suppresses all other overlapping boxes whose IoU exceeds the NMS threshold (e.g. 0.45). This leaves exactly one bounding box per physical pothole.

**Q5: How does your video detection pipeline work?**
> Frames are decoded sequentially via OpenCV `VideoCapture`. Each frame is resized to $416 \times 416$ and evaluated by YOLOv8n. Bounding boxes are filtered by confidence and drawn with severity color coding. Processed frames are re-encoded to H.264 MP4 using `imageio` with `libx264` for browser playback, operating at ~55 FPS.

---

## M6 — Explainability (Grad-CAM)

### What was built
In-house manual Grad-CAM implementation (`src/explainability/gradcam.py`) without third-party black-box libraries; failure gallery generation.

### Mathematical Formulation
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{k, i, j}}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### Viva Q&A

**Q1: What does the Grad-CAM heatmap tell you, and what are its limitations?**
> It visualizes which spatial regions contributed positively to the classifier's class score. Its critical limitation is **coarse spatial resolution**: because MobileNetV2's final feature map is $4 \times 4$, Grad-CAM reveals broad regions of interest but cannot delineate exact pothole boundaries.

**Q2: Why is the ReLU function mandatory in Grad-CAM?**
> Channels with negative gradients suppress the class score (they are evidence *against* the pothole hypothesis). Applying ReLU discards negative contributions, isolating only visual evidence that actively supports the prediction.

**Q3: Why target the LAST convolutional layer?**
> Earlier convolutional layers capture generic edges and textures shared across all images. The final convolutional layer preserves the highest-level semantic representations while still retaining 2D spatial dimensions before global pooling.

**Q4: What failure patterns did you observe in the failure gallery?**
> 8 test misclassifications revealed two primary failure modes:
> - **False Positives:** Tree shadows and dark asphalt sealant patches have sharp contrast gradients mimicking pothole edges.
> - **False Negatives:** Shallow potholes and road glare wash out edge gradients, causing the network to miss low-contrast defects.

**Q5: How does Grad-CAM differ from Grad-CAM++ and Score-CAM?**
> Grad-CAM uses first-order gradients weighted by global average pooling. Grad-CAM++ uses second- and third-order gradients to better capture multiple instances. Score-CAM is gradient-free, weighting activation maps by the model's confidence increase when passing the masked image forward.

---

## M7 — Metric Learning & Report Deduplication (Siamese Network)

### What was built
`siamese.py` — Siamese network with shared CNN backbone; `train_siamese.py` — Contrastive Loss training with t-SNE embedding visualization.

### Contrastive Loss Formulation
$$\mathcal{L}(e_1, e_2, y) = (1 - y) \frac{1}{2} d^2 + y \frac{1}{2} \max(0, m - d)^2$$
where $d = \|e_1 - e_2\|_2$, $y = 0$ for identical potholes, $y = 1$ for different scenes, and margin $m = 1.0$.

### Performance
- **Pair Classification Accuracy:** 96.12%
- **Mean Intra-Class Distance (Same Pothole):** 0.2210 (tightly clustered below threshold $d < 0.5$)
- **Mean Inter-Class Distance (Different Scenes):** 1.1988 (separated beyond margin $m=1.0$)

### ⚠️ Limitation (Mandatory Viva Disclosure)
> Real-world multi-temporal road surveys (the same pothole photographed weeks apart) were unavailable. Positive pairs were synthesized via heavy photometric and spatial augmentations of the same image. The network learns invariant metric representations under affine transformations, but true temporal validation remains future work.

### Viva Q&A

**Q1: Why apply L2 normalization to embeddings?**
> L2 normalization projects all feature vectors onto the surface of a unit hypersphere ($\|e\|_2 = 1$). This bounds Euclidean distances to $[0, 2]$, ensuring the contrastive margin ($m=1.0$) applies uniformly across all training samples.

**Q2: What is weight sharing and why is it essential?**
> Both images pass through the exact same PyTorch `nn.Module` instance. This guarantees that identical feature extractors are applied to both inputs, ensuring distances reflect semantic differences rather than branch-specific transformations.

**Q3: How does this help a municipal complaint portal?**
> When a citizen uploads a pothole photo, its 128-dimensional embedding is compared via Euclidean distance against existing open tickets. If $d < 0.50$, it is flagged as a duplicate submission, merged with the existing ticket, and increments its repair urgency.

**Q4: What is the purpose of the contrastive margin $m$?**
> Without the margin, the loss would attempt to push different-class pairs infinitely far apart, leading to gradient explosion. The margin establishes an equilibrium: once different pairs are at least distance $m$ apart, their loss becomes zero, focusing gradients on hard negative pairs.

**Q5: What does the t-SNE plot validate?**
> t-SNE projects 128-dimensional embeddings to 2D. Clear cluster separation between normal road surfaces and pothole instances confirms that the metric space captures class-discriminative geometry.

---

## M8 & M5 — Negative Results Analysis (Academic Integrity)

### Negative Result 1: Discarded Pseudo-Mask U-Net
- **What was attempted:** Training `SmallUNet` with BCE+Dice loss on rasterized bounding boxes.
- **Why it failed:** Rasterized bounding boxes taught the U-Net to predict rectangular boxes. The 0.9051 Dice score was an artifact of box overlap.
- **The fix:** Discarded U-Net; replaced with YOLOv8n object detection.

### Negative Result 2: Failed ConvAutoencoder Anomaly Detection
- **What was attempted:** Training an unsupervised Autoencoder on normal roads to flag potholes via reconstruction MSE.
- **The result:** Test ROC-AUC of **0.3008** (substantially worse than random guessing).
- **Why it failed:** Natural road surfaces have high-frequency aggregate textures, shadows, and lane paint that produce higher reconstruction errors than dark, smooth pothole cavities. Generative reconstruction is unsuited for unconstrained outdoor textures.

### Viva Q&A

**Q1: Why report an Autoencoder with an AUC of 0.3008?**
> Scientific integrity requires reporting negative results. It demonstrates why unsupervised generative anomaly detection fails on high-variance outdoor imagery, justifying why supervised discriminative models (YOLO and MobileNetV2) are necessary.

**Q2: Why was Dice 0.9051 on U-Net misleading?**
> Dice measures pixel overlap. When ground truth is a rectangle, a model predicting rectangles achieves high Dice without actually delineating irregular pothole boundaries.

**Q3: Can your system measure pothole depth?**
> No. A single monocular RGB camera cannot recover metric 3D depth without calibrated stereo pairs or LiDAR. Our severity metric is an explicit 2D surface area heuristic ($A_{\text{box}} / A_{\text{img}}$).

---

## M10 — Streamlit Deployment Architecture

### Architecture Overview
The production dashboard is organized into 6 interactive tabs:
1. **📸 Detect (Image):** Real-time YOLOv8n object detection, confidence slider, severity heuristic, side-by-side original vs annotated views, and structured results table.
2. **🎥 Detect (Video):** Video inference engine capping clips to 15 seconds, rendering live detections, FPS throughput, and H.264 browser playback via `imageio`.
3. **🔬 Classifier & Grad-CAM:** Binary classification with coarse activation heatmaps and explicit explainability notices.
4. **📊 Experiments & Benchmarks:** Complete model comparison tables, ablation metrics, PR curves, and prediction galleries.
5. **⚠️ Negative Results & Viva:** Post-mortems of U-Net and Autoencoder failures with 3 essential viva defense Q&As.
6. **🗺️ Road Hazard Map:** Interactive municipal telemetry demo using `st.map`.

### Viva Q&A

**Q1: Why use `@st.cache_resource` for model loading?**
> Streamlit re-executes the entire script on every user interaction (e.g. slider adjustment). `@st.cache_resource` ensures heavy PyTorch and YOLO weights are loaded into memory exactly once and reused, preventing memory leaks and multi-second UI freezes.

**Q2: How do you ensure generated video plays in all browsers?**
> Standard OpenCV `mp4v` produces MPEG-4 Part 2, which modern browsers cannot play in HTML5 `<video>` tags. We use `imageio` with the `libx264` codec, which produces genuine H.264 encoded MP4 files supported across all desktop and mobile browsers.

**Q3: How is video length capped for interactive preview?**
> Video processing inspects `cv2.CAP_PROP_FPS` and caps execution at $\min(\text{total\_frames}, \text{fps} \times 15)$, ensuring video inference completes in under 15 seconds on CPU.

**Q4: What heuristic is used for pothole severity?**
> Bounding box area divided by total image area: Low $< 2\%$, Medium $2\%-8\%$, High $> 8\%$. This provides an immediate triage signal for road maintenance dispatch.

**Q5: What are the primary future engineering extensions?**
> (1) Stereo camera calibration for true metric depth/volume measurement, (2) Edge deployment with INT8 quantization, and (3) Municipal GPS telemetry integration with automated work-order generation.
