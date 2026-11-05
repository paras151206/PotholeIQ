# PotholeIQ: Deep Learning Systems for Autonomous Pothole Detection, Localization, and Road Infrastructure Assessment

**Student:** Paras Panchal  
**Degree:** B.Tech Computer Science and Engineering (Artificial Intelligence and Machine Learning)  
**Course:** Deep Learning  
**Date:** November 2026  
**GitHub Repository:** [https://github.com/paras151206/PotholeIQ](https://github.com/paras151206/PotholeIQ)

---

## Abstract

Automating the visual inspection and triage of municipal road damage requires systems that provide actionable engineering signals: multi-instance localization, repair urgency heuristics, visual interpretability, duplicate submission resolution, and video processing. In this capstone project, we develop **PotholeIQ**, an integrated deep learning framework for road hazard assessment. Moving from image-level classification to spatial bounding-box object detection, our findings illustrate:

1. **Classifier Progression:** A flattened Multi-Layer Perceptron (TinyMLP) establishes an empirical baseline of 72.82% test accuracy ($F_1 = 0.7250$). A 4-stage Custom Convolutional Neural Network eliminates 98.3% of parameters ($25.2\text{M} \to 425\text{K}$) while boosting accuracy to 92.23% ($F_1 = 0.9223$, $\text{ROC-AUC} = 0.9868$). A two-phase fine-tuned MobileNetV2 architecture achieves 98.06% accuracy ($F_1 = 0.9806$, $\text{ROC-AUC} = 0.9985$).
2. **Object Detection (YOLOv8n):** Fine-tuning YOLOv8n on 1,590 verified images (5,684 ground-truth bounding boxes, 387 duplicates removed) yields **61.49% mAP@0.50**, **36.18% mAP@0.50:0.95**, **71.46% Precision**, and **55.34% Recall** on 239 held-out test images at 18 ms/frame inference speed, enabling both static image and driving video detection.
3. **Scientific Integrity & Discarded Models:** We document two key negative results:
   - *Discarded U-Net:* Training on rasterized rectangular bounding boxes caused U-Net to learn the box prior rather than true damage morphology; the high Dice (0.9051) was an artifact of box overlap, justifying its replacement by YOLOv8n.
   - *Failed Autoencoder:* An unsupervised ConvAutoencoder trained on clean road images produced a test ROC-AUC of **0.3008** (worse than random guessing) because asphalt textures and shadows exhibit higher reconstruction variance than pothole cavities.
4. **Explainability & Metric Learning:** In-house Grad-CAM saliency mapping exposes coarse activation areas ($4 \times 4$ feature resolution), while a Siamese network achieves 96.12% accuracy on paired synthetic augmentations.

Non-parametric 95% bootstrap confidence intervals confirm non-overlapping bounds between MLPs and CNNs, establishing a rigorous foundation for university viva defense.

---

## 1. Problem Statement & Engineering Requirements

Road potholes cause catastrophic vehicle suspension failures, traffic congestion, and severe road accidents. Traditional municipal road inspections rely on manual visual patrols or specialized LiDAR survey vans that municipal budgets cannot sustain continuously.

**PotholeIQ** addresses this with an end-to-end computer vision pipeline tailored for public works departments:
- **Detection & Localization:** Predict 2D bounding boxes and confidence scores around every distinct pothole in an image or driving video frame.
- **Severity Triage Heuristic:** Compute the 2D surface area ratio occupied by damage ($A_{\text{box}} / A_{\text{img}}$: Low $<2\%$, Medium $2\%-8\%$, High $>8\%$) as an operational heuristic.
- **Video Stream Inference:** Process driving camera footage frame-by-frame with real-time FPS reporting.
- **Explainability:** Audit model decision boundaries using Grad-CAM saliency maps.
- **Scientific Humility:** Transparently report failures and identify the operational boundaries of deep vision models on outdoor asphalt.

---

## 2. Datasets & Rigorous Partitioning Protocols

### 2.1 YOLO Object Detection Dataset (Bounding Boxes)
- **Source:** Kaggle Potholes Detection YOLOv8 (`anggadwisunarto/potholes-detection-yolov8`)
- **Format:** High-resolution road photography with standard YOLO `.txt` labels (`class x_center y_center width height` normalized to $[0, 1]$).
- **De-duplication:** Full MD5 cryptographic hashing identified and removed **387 duplicate images** to prevent train-test data leakage.
- **Stratified Partition (70% Train / 15% Val / 15% Test):**
  - **Train Split:** 1,113 images (3,972 ground-truth boxes, avg 3.57/image)
  - **Validation Split:** 238 images (825 ground-truth boxes, avg 3.47/image)
  - **Test Split (Held-Out):** 239 images (887 ground-truth boxes, avg 3.71/image)
  - **Leakage Verification:** Disjointness test confirmed zero image hash overlap across all three splits ($\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$).

### 2.2 Binary Classification Dataset
- **Source:** Kaggle Pothole Detection Dataset (`atulyakumar98/pothole-detection-dataset`)
- **Total Images:** 681 real road images (329 Pothole / 352 Normal Road).
- **Partition:** Stratified 70/15/15 split (476 train / 102 val / 103 test) with fixed seed `42`.

---

## 3. Architecture Design & Methodologies

### 3.1 Object Detection via YOLOv8n
YOLOv8n employs a modified CSPDarknet53 backbone with C2f (Cross-Stage Partial with two convolutions) feature extraction modules, a Path Aggregation Network (PANet) feature pyramid, and an anchor-free decoupled detection head.
- **Loss Function:**
  $$\mathcal{L}_{\text{YOLO}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$
  where CIoU aligns bounding box center, aspect ratio, and intersection-over-union, and Distribution Focal Loss (DFL) models coordinate regression uncertainty.
- **Training Protocol:** Fine-tuned from COCO-pretrained weights for 8 epochs on an AMD Ryzen 7 7840HS CPU at image resolution $416 \times 416$ with batch size 16 and AdamW optimizer.

### 3.2 Classifier Progression
1. **TinyMLP Baseline:** Flattened input ($3 \times 128 \times 128 = 49,152$ inputs) through dense layers ($49152 \to 512 \to 128 \to 2$) with Dropout ($p=0.3$), totaling 25.2 million parameters.
2. **CustomCNN (4-Stage):** Convolutions with channel progression $[32, 64, 128, 256]$, Batch Normalization, ReLU, MaxPool, and Global Average Pooling (GAP), reducing parameters to 425,026.
3. **MobileNetV2 Transfer:** Pretrained ImageNet backbone fine-tuned in two phases:
   - Phase 1: Backbone frozen, custom head ($1280 \to 256 \to 2$) trained with Adam ($\text{lr} = 10^{-3}$) for 5 epochs.
   - Phase 2: Top 3 InvertedResidual blocks unfrozen, trained with reduced learning rate ($\text{lr} = 5 \times 10^{-5}$) for 6 epochs.

### 3.3 Explainability via Grad-CAM
Manual implementation of Gradient-weighted Class Activation Mapping:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{k, i, j}}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
Targeting the final convolutional layer of MobileNetV2 ($4 \times 4$ spatial feature maps).

---

## 4. Comprehensive Experimental Results

### 4.1 YOLOv8n Object Detection Performance (Held-Out Test Set)

Evaluated on the **239 held-out test images** (887 ground-truth instances):

| Metric | Score | Practical Significance |
|---|:---:|---|
| **Precision** | **71.46%** | Low false alarm rate; high reliability when flagging road damage |
| **Recall** | **55.34%** | Detects over half of all pothole instances; misses minor/shallow edge cracks |
| **mAP @ 0.50** | **61.49%** | Robust detection accuracy under standard PASCAL VOC IoU criteria |
| **mAP @ 0.50:0.95** | **36.18%** | Rigorous COCO multi-threshold overlap evaluation |
| **Inference Latency** | **18.0 ms / frame** | Real-time throughput (~55 FPS) on CPU |
| **Model Size** | **6.47 MB** | Suitable for low-power edge compute and dashcam deployment |

### 4.2 Primary Classifier Benchmark (103 Test Images)

| Architecture | Parameters | Test Accuracy | Macro Precision | Macro Recall | Macro $F_1$ | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TinyMLP Baseline** | 25,232,258 | 72.82% | 0.7334 | 0.7257 | 0.7250 | 0.8502 |
| **CustomCNN** | **425,026** | **92.23%** | **0.9225** | **0.9228** | **0.9223** | **0.9868** |
| **MobileNetV2** | 2,544,130 | **98.06%** | **0.9806** | **0.9806** | **0.9806** | **0.9985** |

### 4.3 Statistical Analysis & Non-Parametric Bootstrap Confidence Intervals

Evaluating small test splits ($N = 103$) with 1,000 bootstrap resamplings (`results/metrics/bootstrap_ci_summary.json`):

| Model | Mean Test Accuracy | 95% Accuracy CI | Mean Test $F_1$ | 95% $F_1$ CI |
|---|:---:|:---:|:---:|:---:|
| **TinyMLP Baseline** | 72.71% | [64.08%, 81.55%] | 72.19% | [63.58%, 80.68%] |
| **CustomCNN** | 92.29% | [87.38%, 97.09%] | 92.22% | [86.77%, 97.08%] |
| **MobileNetV2** | 98.08% | [95.15%, 100.0%] | 98.06% | [95.02%, 100.0%] |

**Statistical Insight:** The 95% $F_1$ confidence intervals for `CustomCNN` ($[0.8677, 0.9708]$) and `TinyMLP` ($[0.6358, 0.8068]$) are completely non-overlapping. This provides **strong empirical evidence** that convolutional feature hierarchies outperform flattened dense projections. However, because the test partition contains 103 images, fine-grained differences between MobileNetV2 and CustomCNN should be stated with appropriate statistical caution.

### 4.4 CustomCNN Ablation Study

| Ablation Variant | Test Accuracy | Macro $F_1$ | ROC-AUC | Observation |
|---|:---:|:---:|:---:|---|
| **Full CustomCNN** | **92.23%** | **0.9223** | **0.9868** | Reference baseline |
| Without Augmentation | 86.41% | 0.8637 | 0.9438 | **-5.86% $F_1$** drop; overfits training road surfaces |
| Without BatchNorm | 88.35% | 0.8833 | 0.9453 | **-3.90% $F_1$** drop; slower optimization convergence |
| Without Dropout | 92.23% | 0.9223 | 0.9872 | Identical test accuracy, but wider train-val loss gap |

*Limitation:* With 103 test samples, subtle regularizers like Dropout fall within the bootstrap margin of error.

---

## 5. Negative Results & Scientific Integrity

A core contribution of this capstone is candidly documenting where theoretical methods broke down in practice:

### 5.1 Discarded U-Net Pseudo-Mask Segmentation
- **Initial Hypothesis:** A Small U-Net trained on rasterized bounding boxes could predict continuous damage masks and inform repair urgency.
- **Empirical Reality:** Because ground truth was constructed by rasterizing rectangular boxes, the U-Net learned the geometric rectangular box prior rather than true ragged pothole contours.
- **Artificial Metrics:** The high reported Dice coefficient ($0.9051$) and IoU ($0.8276$) were mathematically inflated artifacts of predicting rectangles to match rectangles.
- **Engineering Decision:** Discarded U-Net from the main inference pipeline. Object detection (YOLOv8n) is the mathematically correct paradigm for bounding box supervision.

### 5.2 Failed Autoencoder Anomaly Detection
- **Initial Hypothesis:** Training an unsupervised Convolutional Autoencoder on normal road images would detect potholes via elevated reconstruction MSE.
- **Empirical Reality:** Evaluated on the test split, the autoencoder achieved an ROC-AUC of **0.3008** (substantially worse than random chance).
- **Failure Cause:** Natural road surfaces exhibit high-frequency aggregate textures, shadows, tar lines, and white lane paint that produce higher reconstruction errors than smooth dark pothole basins. Unsupervised generative reconstruction failed on unconstrained outdoor asphalt.

### 5.3 Synthetic Siamese Pairs
- **The Caveat:** Metric learning with Contrastive Loss achieved 96.12% pair accuracy, but positive pairs were synthesized via heavy spatial and photometric augmentations due to the absence of multi-temporal road surveys in public datasets.

---

## 6. Software Architecture & Deployment

The system is packaged as an interactive Streamlit application (`app/app.py`):
- **Tab 1 (Detect Image):** Real-time YOLOv8n object detection with confidence slider, severity heuristic, side-by-side view, and structured bounding-box coordinates table.
- **Tab 2 (Detect Video):** Video processing pipeline capping clips to 15 seconds, rendering live detection counts, FPS throughput, and H.264 browser-compatible video playback via `imageio`.
- **Tab 3 (Classifier & Grad-CAM):** Binary classification with coarse activation heatmaps and explicit explainability notices.
- **Tab 4 (Experiments):** Full model comparison tables, ablation metrics, PR curves, and prediction galleries.
- **Tab 5 (Negative Results & Viva):** Transparent post-mortem of U-Net and Autoencoder failures with 3 essential viva defense Q&As.
- **Tab 6 (Road Hazard Map):** Interactive demo map with simulated GPS damage points.

---

## 7. Viva Defense Q&A Guide

**Q1: Why did you transition from U-Net segmentation to YOLOv8n object detection?**  
*Answer:* The dataset annotations were bounding boxes. Rasterizing them into rectangular masks causes a segmentation network to memorize rectangular bounding priors rather than irregular pothole contours. Object detection is the mathematically proper formulation for bounding-box supervision.

**Q2: What is the significance of your 95% bootstrap confidence intervals?**  
*Answer:* With a 103-image test split, single point metrics carry sampling variance. 1,000 bootstrap resamplings showed zero overlap between CNN ($[86.77\%, 97.08\%]$) and MLP ($[63.58\%, 80.68\%]$), establishing strong empirical evidence of superior convolutional inductive bias.

**Q3: Can your system measure pothole depth or volume?**  
*Answer:* No. A monocular RGB camera cannot recover metric depth without stereo calibration or LiDAR. Our severity index is an explicit 2D surface area heuristic.

---

## References

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0). *GitHub*.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*, 4510–4520.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*, 618–626.
4. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, 234–241.
