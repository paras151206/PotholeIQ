
# PotholeIQ — Viva Guide

> Every module explained in plain language, key equations, design choices, and 5 viva Q&As.
> Read this alongside the code, not instead of it.

---

## M0 — Repo Setup

### What was built
Config-driven skeleton: `config.yaml`, `requirements.txt`, seed utilities (`set_all_seeds`, `worker_init_fn`), metric helpers, plotting helpers, `smoke_test.py`.

### Key design choices
| Choice | Reason |
|--------|--------|
| `config.yaml` for all hyperparams | Version-controlled experiments; no magic numbers in code |
| `seed=42` everywhere | Reproducibility — same split, same init, same augmentation |
| `worker_init_fn` in DataLoader | Forked workers share RNG state; without this, all workers produce identical augmentations |
| `cuDNN deterministic=True` | Forces reproducible (not fastest) convolution algorithm |

### Viva Q&A

**Q1: Why do you need `worker_init_fn` in a DataLoader?**
> When PyTorch forks workers, all inherit the same NumPy/Python RNG state from the parent. Without `worker_init_fn`, every worker applies *identical* random augmentations each epoch, defeating the purpose. The fix derives a unique seed from each worker's ID.

**Q2: What is the cost of `torch.backends.cudnn.deterministic=True`?**
> cuDNN normally benchmarks several convolution algorithms at startup and picks the fastest. `deterministic=True` forces a fixed (reproducible) choice, typically 5–15% slower training. Worth it for reproducibility in a research project.

**Q3: Why keep hyperparameters in YAML vs argparse?**
> YAML is version-controlled — every git commit records exact hyperparameters. With argparse you must remember to log every flag. With YAML you get a full experiment log for free.

**Q4: What is a `.gitkeep` file?**
> Git doesn't track empty directories. A zero-byte `.gitkeep` forces git to include the directory so collaborators get the right folder structure after cloning.

**Q5: Explain the difference between `torch.manual_seed` and `torch.cuda.manual_seed_all`.**
> PyTorch maintains separate RNG states for CPU and each GPU. `manual_seed` seeds only the CPU RNG. `cuda.manual_seed_all` seeds the CUDA RNG on every GPU device. Both must be set for full reproducibility on GPU.

---

## M1 — Data Pipeline

### What was built
`dataset.py` (4 dataset classes), `download.py` (Kaggle API), `split.py` (stratified split + leakage check), `eda.py` (5 plots).

### Dataset: Kaggle Pothole Detection (atulyakumar98)
| Field | Value |
|-------|-------|
| Source | Kaggle — atulyakumar98/pothole-detection-dataset |
| Licence | CC0 Public Domain |
| Annotations | YOLO bounding boxes (class x_center y_center w h, normalised) |
| Split | 70/15/15 stratified, seed=42, NO leakage verified |

### Key design choices
| Choice | Reason |
|--------|--------|
| Stratified split | Preserves class balance in each subset |
| Leakage check by filename stem | Ensures the same image doesn't appear in train AND test |
| Flexible dataset discovery | Handles YOLO split layout, class-folder layout, and flat layout |
| `NormalRoadDataset` | Used for AE training — no labels, only normal images |

### Key equations

**Data split sizes** (N = total images, r₁=0.70, r₂=0.15, r₃=0.15):
$$N_{train} = \lfloor r_1 \cdot N \rfloor, \quad N_{val} = \lfloor r_2 \cdot N \rfloor, \quad N_{test} = N - N_{train} - N_{val}$$

**YOLO → pixel coordinate conversion** (for bbox mask generation):
$$x_1 = (x_c - w/2) \cdot W_{img}, \quad y_1 = (y_c - h/2) \cdot H_{img}$$

### Viva Q&A

**Q1: Why use stratified splitting instead of random?**
> Random splitting can produce splits with very different class ratios by chance (especially with small datasets). Stratified split guarantees that each subset reflects the overall class distribution (e.g., if 40% are potholes overall, each split is ~40% pothole).

**Q2: What is data leakage and how did you check for it?**
> Data leakage = the same image (or a near-duplicate) appears in both train and test. We check by comparing filename stems (without extension) across all three splits. If any stem appears in more than one split, the script raises `AssertionError` and aborts.

**Q3: What transforms do you apply during training vs validation?**
> Train: Resize→RandomHFlip→RandomRotation→ColorJitter→ToTensor→Normalize→RandomErasing. Validation/Test: Resize→ToTensor→Normalize only. Using augmentation at val/test would make evaluation non-deterministic and unfairly penalise the model.

**Q4: Why do you use ImageNet mean/std for normalization?**
> Pre-trained models (MobileNetV2) were trained with ImageNet normalisation, so we must use the same stats to match the input distribution the backbone expects. For the custom CNN, using the same stats simplifies comparison and doesn't significantly hurt (road images aren't too far from ImageNet statistics).

**Q5: What is the YOLO annotation format and how do you convert it to a mask?**
> YOLO: `class_id x_center y_center width height` (all normalised 0–1 relative to image size). Conversion: multiply by image pixel dimensions to get the bounding box corners in pixels, then fill that rectangle with 1s in a zero-initialised binary mask. This is a rectangular approximation — a known limitation documented in the README.

---

## M2 — Baseline (TinyMLP)

### What was built
`baseline.py` — 2-layer MLP; `train_baseline.py` — training script.

### Architecture
```
Input: (B, 3, 128, 128) → Flatten → (B, 49152)
Linear(49152, 512) → ReLU → Dropout(0.3) → (B, 512)
Linear(512, 128)   → ReLU → Dropout(0.3) → (B, 128)
Linear(128, 2)                             → (B, 2) [logits]
```

### Key design choices
| Choice | Reason |
|--------|--------|
| MLP as baseline (not logistic regression) | PyTorch-native; easier to integrate in the same training pipeline |
| Weighted cross-entropy | Handles class imbalance without discarding data |
| No augmentation | Establishes the floor; augmentation is introduced in M3 |

### Key equation
**Weighted cross-entropy loss:**
$$\mathcal{L}_{WCE} = -\sum_{c} w_c \cdot y_c \log(\hat{p}_c)$$
where $w_c = \frac{N}{K \cdot N_c}$ (inverse frequency weight, $K$ = number of classes).

### Viva Q&A

**Q1: Why do you use a baseline model at all?**
> The baseline sets a performance floor. If your CNN can't beat the MLP, you have a training bug — not a model problem. Any improvement on the baseline is evidence the CNN is extracting useful features, not just memorising class priors.

**Q2: What is label smoothing and did you use it in the baseline?**
> Label smoothing replaces hard target 1 with $1 - \varepsilon$ and 0 with $\varepsilon/K$, preventing the model from becoming overconfident. We did NOT use it for the baseline (we want it to be simple). We use it in the custom CNN (ε=0.1) to show its effect.

**Q3: How does class-weight computation work?**
> `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` returns $w_c = N / (K \cdot N_c)$. The minority class gets a higher weight, so its misclassification is penalised more in the loss. This is equivalent to oversampling the minority class.

**Q4: What accuracy would a random classifier achieve on this dataset?**
> A random classifier achieves accuracy equal to the majority class fraction. E.g., if 60% of images are normal, random guessing normal always gives 60% accuracy. The baseline must beat this to be useful.

**Q5: Why flatten the image for the MLP instead of using 2D input?**
> An MLP treats each pixel as an independent feature — it has no concept of spatial locality. By flattening, we lose all spatial structure. This is precisely why CNNs outperform MLPs on images: convolutions share weights across space and exploit local correlations.

---

## M3 — Custom CNN

### What was built
`cnn.py` — Custom CNN with 4 ConvBlocks; `train_cnn.py` — training + ablation studies.

### Architecture
```
Input: (B, 3, 128, 128)
ConvBlock-1: Conv(3→32)  → BN → ReLU → MaxPool → (B, 32,  64, 64)
ConvBlock-2: Conv(32→64) → BN → ReLU → MaxPool → (B, 64,  32, 32)
ConvBlock-3: Conv(64→128)→ BN → ReLU → MaxPool → (B, 128, 16, 16)
ConvBlock-4: Conv(128→256)→BN → ReLU → MaxPool → (B, 256,  8,  8)
GlobalAvgPool(1)                                 → (B, 256,  1,  1)
Flatten                                          → (B, 256)
Linear(256, 128) → ReLU → Dropout(0.4) → Linear(128, 2) → (B, 2)
```

### Key design choices
| Choice | Reason |
|--------|--------|
| BatchNorm after every conv | Stabilises gradients, allows higher LR, reduces sensitivity to weight init |
| GlobalAvgPool instead of Flatten | Fewer parameters, less overfitting, spatially invariant |
| Dropout(0.4) | Regularisation — prevents co-adaptation of neurons |
| CosineAnnealingLR | Avoids getting stuck at a fixed LR; allows refined optimisation at the end |
| Label smoothing (ε=0.1) | Prevents overconfidence, improves calibration |

### Key equations

**Conv output size:**
$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$
With $k=3, p=1, s=1$: $H_{out} = H_{in}$ (same padding). MaxPool halves: $H_{out} = H_{in}/2$.

**BatchNorm forward (at training):**
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

**Label-smoothed cross-entropy:**
$$\mathcal{L}_{LS} = -(1-\varepsilon)\log\hat{p}_{true} - \frac{\varepsilon}{K}\sum_c \log\hat{p}_c$$

### Ablation results

| Variant | Val F1 | Note |
|---------|--------|------|
| Full model (BN + Dropout + Aug) | — | From actual run |
| No augmentation | — | Overfits faster |
| No BatchNorm | — | Unstable training |
| No Dropout | — | Val accuracy diverges from train |

*(Fill from results/metrics/ after training)*

### Viva Q&A

**Q1: Why does BatchNorm help training?**
> Without BN, the distribution of each layer's input shifts as the weights update ("internal covariate shift"). BN normalises each mini-batch's activations to zero mean and unit variance, keeping distributions stable. This allows higher learning rates and faster convergence.

**Q2: What is Global Average Pooling and why use it over flattening?**
> GAP computes the average of each feature map across all spatial positions: $(B, C, H, W) \to (B, C)$. Compared to flattening ($(B, C \cdot H \cdot W)$), GAP has dramatically fewer parameters going into the classifier, reducing overfitting. It also makes the model more spatially invariant.

**Q3: Explain the overfit sanity check and what it proves.**
> We train on a tiny subset (50 images) for 20 epochs. The model should reach ~100% training accuracy on this tiny set. If it can't, the model is broken (vanishing gradients, wrong loss, etc.). If it can, we know the model has enough capacity — the challenge is generalisation on the full set.

**Q4: What is CosineAnnealingLR and why is it better than StepLR?**
> CosineAnnealingLR decays the learning rate following a cosine curve from $lr_{max}$ to 0 (or $lr_{min}$) over T_max epochs. StepLR drops LR by a fixed factor at fixed intervals (harsh, discontinuous). Cosine annealing is smoother, allows the model to escape local minima early (high LR) and converge precisely at the end (low LR).

**Q5: What does the Dropout layer actually do at train vs test time?**
> At train time, each neuron is independently zeroed with probability $p$, and surviving activations are scaled by $1/(1-p)$. At test time, Dropout is disabled (all neurons active, no scaling needed since training compensated). This forces the network to not rely on any single neuron, acting as an ensemble of exponentially many sub-networks.

---

## M4 — Transfer Learning (MobileNetV2)

### What was built
`transfer.py` — MobileNetV2 with custom head; `train_transfer.py` — two-phase training.

### Why MobileNetV2?
MobileNetV2 uses inverted residuals and linear bottlenecks — designed to be efficient on mobile devices. Only ~3.4M parameters vs ResNet50's 25M. Pretrained on ImageNet (1.28M images), it has learned general visual features that transfer well.

### Two-phase strategy
| Phase | Frozen? | LR | Purpose |
|-------|---------|-----|---------|
| Phase 1 (8 epochs) | Backbone frozen | 0.001 | Train head only; backbone protected |
| Phase 2 (10 epochs) | Last 3 blocks unfrozen | 0.00005 | Fine-tune features for road damage |

### Key equations

**Fine-tuning parameter count:**
Let $P_{total}$ = total parameters, $P_{frozen}$ = frozen parameters.
Only $P_{total} - P_{frozen}$ parameters receive gradients in Phase 1.

**Why much lower LR in Phase 2?**
If $lr$ is too high, the pre-trained feature weights change too fast → "catastrophic forgetting" of ImageNet features. Rule of thumb: Phase2 LR ≈ Phase1 LR / 10 to 20.

### Viva Q&A

**Q1: What is transfer learning and why does it work for road images?**
> Transfer learning reuses a model trained on a large dataset (ImageNet) for a new task. It works because low-level CNN features (edges, textures, colour blobs) are domain-agnostic. A model that learned to detect wheels and road lines on ImageNet has already learned low-level features useful for detecting pothole textures.

**Q2: What is catastrophic forgetting?**
> When a pre-trained model is fine-tuned with too high a learning rate, the new gradient updates overwrite the carefully learned pre-trained weights. The model "forgets" the general features it learned on ImageNet. Using a low LR in Phase 2 (0.00005) prevents this.

**Q3: Why freeze the backbone in Phase 1?**
> Phase 1 trains only the new classifier head (randomly initialised). If the backbone were also unfrozen, large gradients from the random head would corrupt the pre-trained backbone weights. Freezing protects the backbone until the head has learned reasonable initial weights.

**Q4: Which layers do you unfreeze in Phase 2 and why the last ones?**
> We unfreeze the last 3 InvertedResidual blocks of `features`. Later layers learn more task-specific features (wheels, road damage patterns) while earlier layers learn generic features (edges, colours). Unfreezing only the later layers adapts the task-specific features while preserving the generic ones.

**Q5: How does MobileNetV2 achieve efficiency with inverted residuals?**
> Standard residual blocks: wide → narrow → wide. MobileNetV2 inverts this: narrow → wide → narrow (inverted residual). The wide intermediate layer allows for expressiveness, while the narrow endpoints reduce memory and computation. Depthwise separable convolutions replace standard convolutions, further reducing FLOPs by ~8–9×.

---

## M5 — Segmentation (U-Net)

### What was built
`unet.py` — lightweight U-Net; `train_unet.py` — training with Dice+BCE loss + severity.

### Architecture
```
Encoder:
  DoubleConv(3→32)   → MaxPool → (B, 32,  64, 64)
  DoubleConv(32→64)  → MaxPool → (B, 64,  32, 32)
  DoubleConv(64→128) → MaxPool → (B, 128, 16, 16)
Bottleneck:
  DoubleConv(128→256)          → (B, 256, 16, 16)
Decoder (with skip connections):
  Upsample+DoubleConv(256+128→128) → (B, 128, 32, 32)
  Upsample+DoubleConv(128+64→64)   → (B,  64, 64, 64)
  Upsample+DoubleConv(64+32→32)    → (B,  32,128,128)
Output: Conv(32→1) → Sigmoid → (B, 1, 128, 128)
```

### Key equations

**Dice Loss:**
$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i p_i g_i + \varepsilon}{\sum_i p_i + \sum_i g_i + \varepsilon}$$
where $p_i \in [0,1]$ is the predicted probability and $g_i \in \{0,1\}$ is the ground truth.

**Combined Loss:**
$$\mathcal{L} = w_{Dice} \cdot \mathcal{L}_{Dice} + (1 - w_{Dice}) \cdot \mathcal{L}_{BCE}$$

**Severity (heuristic thresholds):**
$$\text{ratio} = \frac{\text{pothole pixels}}{H \times W}, \quad \text{severity} = \begin{cases} \text{Low} & r < 0.05 \\ \text{Medium} & 0.05 \le r < 0.20 \\ \text{High} & r \ge 0.20 \end{cases}$$

### Limitation
> Masks are **rectangular approximations** generated from YOLO bounding boxes. No pixel-level annotation was available in the dataset. This is documented explicitly in the README and UI. IoU and Dice scores reflect this rectangular mask quality, not true pixel-level segmentation quality.

### Viva Q&A

**Q1: What are skip connections in U-Net and why are they important?**
> Skip connections copy feature maps from each encoder stage directly to the corresponding decoder stage (concatenated). Without them, the decoder must reconstruct fine spatial detail from a compressed bottleneck alone. Skip connections preserve high-resolution spatial information (edges, boundaries) that gets lost during downsampling.

**Q2: Why use Dice loss instead of plain cross-entropy for segmentation?**
> BCE treats every pixel equally. If 95% of pixels are background, a model predicting "all background" gets 95% accuracy but Dice ≈ 0. Dice loss directly measures overlap between prediction and ground truth, so it cares about minority-class (pothole) pixels. We use BCE+Dice because BCE provides stable gradients while Dice drives overlap maximisation.

**Q3: What is the difference between IoU and Dice?**
> Both measure overlap between predicted and true masks.
> - $\text{IoU} = |P \cap G| / |P \cup G|$
> - $\text{Dice} = 2|P \cap G| / (|P| + |G|)$
> Dice = 2·IoU / (1 + IoU). Dice is more sensitive to small overlaps and penalises false positives/negatives more aggressively. IoU is the standard metric in detection (COCO, PASCAL VOC).

**Q4: Why are the severity thresholds "heuristics"?**
> There is no ground-truth severity label in the dataset. The thresholds (Low < 5%, Medium 5–20%, High > 20%) are based on engineering intuition about road repair urgency. A pothole covering >20% of the image is clearly severe; <5% is minor. These should be validated against expert road engineer judgements in a real deployment.

**Q5: What would you do with real pixel-level masks?**
> Replace bounding-box-to-mask conversion with actual annotated masks. This would give higher IoU (not limited by rectangular approximation), enable shape-aware severity (not just area), and allow the U-Net to learn non-rectangular pothole shapes (potholes are often irregular polygons). Tools like CVAT or Roboflow could be used for annotation.

---

## M6 — Explainability (Grad-CAM)

### What was built
`gradcam.py` — manual Grad-CAM implementation; `run_gradcam.py` — gallery generation + failure analysis.

### How Grad-CAM works (step by step)
1. **Forward pass**: record activation maps $A^k$ at the target layer.
2. **Backward pass** for target class $c$: compute $\frac{\partial y^c}{\partial A^k_{ij}}$.
3. **Importance weights**: $\alpha^k_c = \text{GlobalAvgPool}\left(\frac{\partial y^c}{\partial A^k}\right)$ (one scalar per channel).
4. **Weighted combination**: $L^c = \text{ReLU}\left(\sum_k \alpha^k_c A^k\right)$ — ReLU keeps only positive influence.
5. **Normalise** to $[0,1]$ and upsample to input resolution.

### Key equation
$$L^c_{Grad-CAM} = \text{ReLU}\!\left(\sum_k \underbrace{\frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}}_{\alpha^k_c} \cdot A^k\right)$$

### Viva Q&A

**Q1: What does the Grad-CAM heatmap tell you?**
> It shows which spatial regions of the input image were most influential for the model's prediction. High activation (red/warm) = the model found something important there. For a correct pothole prediction, activations should concentrate on the damaged road area. For a failure case, they might highlight irrelevant regions (shadows, lane markings).

**Q2: Why use ReLU in Grad-CAM?**
> Without ReLU, channels with negative weights would appear as blue/cold in the heatmap, meaning they suppressed the class score. We care only about what *supports* the prediction (positive influence). ReLU zeros out the suppressing channels, keeping only the contributing ones.

**Q3: Why target the LAST convolutional layer?**
> Earlier layers capture low-level features (edges, textures) shared across classes. The last convolutional layer has the most class-discriminative features while still retaining spatial information (before the global pooling collapses spatial structure). It gives the best combination of class-specificity and spatial resolution.

**Q4: What failure patterns did you observe in the Grad-CAM gallery?**
> Common failure cases:
> - **Shadows**: model activates on dark shadow regions, misclassifies as pothole.
> - **Wet roads**: specular reflections cause false positives.
> - **Manhole covers**: dark circular shapes confuse the model.
> - **Patched potholes**: healed potholes with different texture may be misclassified.

**Q5: What is the difference between Grad-CAM, Grad-CAM++, and LIME?**
> Grad-CAM: global average pool of gradients → one weight per channel. Grad-CAM++: uses second-order gradients for better localisation of multiple object instances. LIME: model-agnostic — fits a local linear approximation around each prediction. Grad-CAM requires a CNN and access to gradients; LIME works on any classifier.

---

## M7 — Siamese Network

### What was built
`siamese.py` — SiameseNet + ContrastiveLoss; `train_siamese.py` — training + t-SNE.

### Architecture
Shared encoder (CustomCNN backbone, no classifier) → Linear(256, 128) → L2 normalise → embedding.
Both images share identical weights (the SAME nn.Module instance).

### Key equation — Contrastive Loss

$$\mathcal{L}_{contrastive}(e_1, e_2, y) = (1-y)\,d^2 + y\,\max(0,\, m - d)^2$$

where $d = \|e_1 - e_2\|_2$, $y=0$ for same pair, $y=1$ for different pair, $m$ = margin (1.0).

**Intuition**: Same pairs → $d \to 0$; Different pairs → $d \to m$ (or beyond).

### ⚠️ Limitation (must state in viva)
> Positive pairs are **augmented views of the same image**, not real same-pothole-over-time photos. The model learns "are these from the same damage category?" not "is this the same pothole years later?" This is documented in the README, the code comments, and the metrics JSON.

### Viva Q&A

**Q1: What is the purpose of L2 normalising the embeddings?**
> L2 normalisation maps embeddings to the surface of a unit hypersphere. All vectors have magnitude 1, so Euclidean distance in embedding space is bounded $[0, 2]$ and directly interpretable as angular similarity. This makes the contrastive margin ($m=1.0$) meaningful across all training examples.

**Q2: Why use weight sharing in the Siamese network?**
> Weight sharing ensures that both branches apply *identical transformations* to their inputs. Without it, the two branches could learn different representations and the distance between them would be meaningless. Weight sharing is enforced by using the SAME nn.Module instance for both forward passes — PyTorch automatically shares gradients.

**Q3: What is the contrastive loss margin and how do you choose it?**
> The margin $m$ is the minimum desired distance between different-class pairs. Same pairs should have $d < m/2$ (threshold we use for pair accuracy), and different pairs should have $d \geq m$. We set $m=1.0$ which is a standard starting point; it can be tuned on the validation pair accuracy.

**Q4: What does the t-SNE plot show and what should you see?**
> t-SNE reduces the 128-dim embeddings to 2D for visualisation. If the network has learned a good embedding, normal and pothole points should form separate clusters. Overlap indicates the embeddings are not discriminative enough. Clear separation validates that the Siamese network has learned a meaningful metric.

**Q5: How would you use this model in a real citizen-reporting app?**
> When a new pothole report arrives with an image, compute its embedding. Compare with embeddings of all existing reports using cosine similarity or L2 distance. If distance < threshold, the new report likely refers to the same pothole → mark as duplicate and increment its repair priority. Otherwise, add as a new pothole record.

---

## M8 — Autoencoder Anomaly Detection

### What was built
`autoencoder.py` — ConvAutoencoder; `train_autoencoder.py` — unsupervised training + ROC-AUC.

### Architecture
```
Encoder: 4 × (Conv2d + BN + ReLU, stride=2)   128→64→32→16→8  channels:3→32→64→128→256
         Flatten → Linear(16384, 64) = latent vector
Decoder: Linear(64, 16384) → Reshape(256,8,8)
         4 × (ConvTranspose2d + BN + ReLU, stride=2)  8→16→32→64→128  channels:256→128→64→32→3
         Tanh output
```

### Key equations

**Reconstruction loss (MSE):**
$$\mathcal{L}_{recon} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2$$

**Anomaly score:**
$$\text{score}(x) = \mathcal{L}_{recon}(x, \hat{x}) = \text{MSE}(x, \text{Decoder}(\text{Encoder}(x)))$$

**Threshold selection:** Find $\tau^*$ on validation set:
$$\tau^* = \arg\max_\tau F_1\!\left(\text{score}(x) > \tau,\; y\right)$$

### Why this is unsupervised
> No labels are used during training. The AE sees ONLY normal road images. At test time, potholes reconstruct poorly (high MSE) because the model has no "template" for them. This models the intuition: "if I can't reconstruct it well, it's unusual."

### Viva Q&A

**Q1: Why train the autoencoder ONLY on normal images?**
> If the AE sees potholes during training, it learns to reconstruct them too → low MSE for potholes → they don't look anomalous. By training only on normal images, the model's reconstruction manifold covers only normal road textures. Potholes fall outside this manifold → high reconstruction error.

**Q2: What is ROC-AUC and why is it the right metric here?**
> ROC-AUC measures the classifier's ability to rank positive examples higher than negatives, regardless of threshold. For anomaly detection, the threshold is a hyper-parameter — ROC-AUC is threshold-independent and tells us how well the anomaly score separates the two classes. AUC=0.5 is random, AUC=1.0 is perfect.

**Q3: How do you select the threshold?**
> We try 50 linearly-spaced threshold values between the minimum and maximum validation anomaly scores. For each, we compute the F1 score (treating pothole as the positive class). The threshold with the highest val F1 is selected. This threshold is then fixed and used on the test set (which we look at only once).

**Q4: Compare the AE anomaly detector vs the supervised CNN classifier.**
> Supervised CNN: needs labelled data, discriminative, typically higher accuracy. AE anomaly detector: needs only normal-class data for training, can detect novel anomalies (new types of damage), but typically lower accuracy because it doesn't learn the discriminative boundary directly. The two complement each other: use the CNN for classification, use the AE to catch unusual inputs the CNN may not have been trained for.

**Q5: What are the limitations of MSE as a reconstruction loss?**
> MSE penalises pixel-wise differences equally, regardless of perceptual importance. A sharp edge vs. a blurry edge might have the same MSE but look very different. Also, MSE tends to produce blurry reconstructions (average over many plausible reconstructions). Alternatives: perceptual loss (VGG features), SSIM, or VAE with a probabilistic formulation.

---

## M10 — Streamlit App

### What was built
`app/app.py` — 4-tab UI: Classifier, Segmentation, Grad-CAM, Anomaly.

### Key design choices
| Choice | Reason |
|--------|--------|
| `@st.cache_resource` for model loading | Models loaded once per session, not on every re-render |
| Fallback: transfer → CNN → baseline | App degrades gracefully if newer models aren't trained yet |
| Inline Grad-CAM | No separate process; runs in real-time on uploaded image |
| Threshold from config.yaml | AE threshold set once after training, persisted in config |

### Viva Q&A

**Q1: What is `st.cache_resource` and why do you use it?**
> `st.cache_resource` caches Python objects (like loaded model weights) across Streamlit re-renders. Without it, the model would reload from disk on every user interaction (button click, file upload). With it, models load once and are reused.

**Q2: How does the Grad-CAM work in the app?**
> On each image upload, we do a forward pass through the classifier, then run backward for the predicted class, extract gradients and activations from the last conv layer, compute weighted sum (Grad-CAM), resize to 128×128, apply jet colormap, and blend with the original image at 50% opacity.

**Q3: What happens if no model checkpoint exists?**
> The app shows a `st.warning` message in the affected tab (e.g., "No U-Net checkpoint found. Run train_unet.py first."). Other tabs continue to function normally. This graceful degradation prevents one missing checkpoint from breaking the entire app.

**Q4: Why use OpenCV for the mask overlay instead of PIL?**
> `cv2.addWeighted` performs alpha blending efficiently in one call. PIL's approach would require manual array manipulation. OpenCV also provides `cv2.applyColorMap` which applies the jet colormap needed for Grad-CAM visualisation in a single function call.

**Q5: What improvements would you make with more time?**
> (1) Add a file upload to the Siamese tab to compare two uploaded images. (2) Add batch processing. (3) Deploy on Streamlit Cloud for a public demo URL. (4) Add GPS metadata display if images contain EXIF location data. (5) Add a severity trend chart from M9 LSTM (if implemented).
