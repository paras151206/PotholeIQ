# PotholeIQ — Project Report

**Course:** Deep Learning | **Semester:** —  
**Student:** — | **Roll No:** —

---

## Abstract

PotholeIQ is an end-to-end deep learning pipeline for pothole detection,
severity estimation, explainability, duplicate detection, and anomaly
scoring from road images.  The system chains a binary classifier (custom
CNN and fine-tuned MobileNetV2), a U-Net segmentation head, Grad-CAM
explainability, a Siamese network for duplicate-report merging, and a
convolutional autoencoder for unsupervised anomaly detection.  All models
are trained on a public road-damage dataset without any paid APIs or
proprietary data.  Results are course-level and are not claimed to be
state-of-the-art.

*(Fill in key numbers after training is complete.)*

---

## 1. Introduction

Road potholes are a significant public-safety and economic hazard.
Automated detection from camera images can accelerate inspection and
prioritise repair.  This project demonstrates how multiple deep learning
techniques—classification, segmentation, explainability, metric learning,
and anomaly detection—can be composed into a single reproducible pipeline.

---

## 2. Method

*(To be completed module by module.)*

### 2.1 Data Preparation (M1)
### 2.2 Baseline (M2)
### 2.3 Custom CNN (M3)
### 2.4 Transfer Learning (M4)
### 2.5 Segmentation (M5)
### 2.6 Explainability (M6)
### 2.7 Siamese Network (M7)
### 2.8 Autoencoder (M8)

---

## 3. Experiments

*(Results tables to be inserted from results/metrics/*.json after runs.)*

---

## 4. Results

---

## 5. Discussion

---

## 6. Conclusion

---

## References

1. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
2. Sandler et al., "MobileNetV2," CVPR 2018.
3. Ronneberger et al., "U-Net," MICCAI 2015.
4. Selvaraju et al., "Grad-CAM," ICCV 2017.
5. Koch et al., "Siamese Neural Networks for One-shot Image Recognition," ICML 2015.

