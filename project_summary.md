# 🧠 NeuroScan AI — Brain Tumor Detection Project
### Complete Work Summary (Presentation Ready)

---

## 1. Project Overview

**NeuroScan AI** is a full-stack AI-powered web application for brain tumor detection from MRI images. It combines a trained deep learning classifier with a traditional computer vision detection pipeline, all served through a polished Flask web interface.

> **Goal**: Upload a brain MRI scan → Get instant AI-powered tumor classification, visual detection, volume estimation, and risk assessment.

---

## 2. Dataset Used

| Property | Detail |
|---|---|
| **Dataset Name** | BraTS 2023 GLI Challenge – Validation Data |
| **Source** | ASNR-MICCAI BraTS 2023 (Synapse Platform) |
| **Format** | NIfTI (`.nii.gz`) 3D MRI volumes |
| **Modality Used** | **T1c** (T1 Contrast-Enhanced) — best for tumor visibility |
| **Total Patients** | **219 patient cases** |
| **"Yes" (Tumor) Images** | **4,341 PNG slices** extracted from real MRI volumes |
| **"No" (No Tumor) Images** | **4,341 PNG slices** (synthetically generated) |
| **Total Images** | **8,682 images** |
| **Actually Used in Training** | **~1,200 images** (600 per class, hardware-capped) |

### How the Dataset Was Prepared
1. Each 3D NIfTI volume was sliced into **2D axial PNG images** (top 15% and bottom 15% of slices skipped — mostly background)
2. Up to **20 informative slices per patient** were saved to `data_slices/yes/`
3. **Synthetic "no-tumor" images** were created by taking real MRI scans and **blacking out the central 40%** (where tumors usually appear), leaving only peripheral brain/skull signal — then adding slight Gaussian noise to prevent the model from cheating
4. Images were capped at **600 per class** for CPU memory efficiency

---

## 3. Model Architecture & Training

### Architecture
| Property | Detail |
|---|---|
| **Model** | ResNet18 (Transfer Learning) |
| **Why ResNet18?** | ~5× lighter than VGG16 — feasible to train on CPU |
| **Pre-trained on** | ImageNet (IMAGENET1K_V1 weights) |
| **Output Layer** | 2-class FC head (tumor / no-tumor) |
| **Input Size** | 96×96 pixels |
| **File** | `model/brain_tumor_resnet18.pth` (~44 MB) |

### Training Pipeline (`train_brats.py`)
```
Stage 1 — Frozen Backbone (10 epochs)
  ├── Only the final FC head is trained
  ├── Optimizer: Adam, LR = 1e-3
  └── Scheduler: CosineAnnealingLR

Stage 2 — Full Fine-Tuning (8 epochs)
  ├── All layers unfrozen
  ├── Optimizer: Adam, LR = 1e-5
  └── Scheduler: CosineAnnealingLR
```

### Data Augmentation Applied
- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation (±15°)
- Color Jitter (brightness ±20%, contrast ±20%)
- ImageNet normalization

### Train / Val Split
- **80% Training** / **20% Validation**
- Best model checkpoint saved automatically based on validation accuracy

---

## 4. Classification System (`classifier.py`)

The classifier runs in **two modes** — automatically selected:

| Mode | When Used | How |
|---|---|---|
| **Model Mode** | When `brain_tumor_resnet18.pth` is present & PyTorch available | Runs image through ResNet18, returns `yes`/`no` + confidence % |
| **Heuristic Mode** | Fallback (no GPU/model required) | OpenCV-based scoring using intensity, variance, contour ratio, and brain asymmetry |

This ensures the app **always works**, even without a trained model or GPU.

---

## 5. Tumor Detection Pipeline (`tumor_detector.py`)

A 6-step OpenCV image processing pipeline that visually locates the tumor in the MRI:

```
Step 1 → Resize to 500×590px
Step 2 → Convert to Grayscale
Step 3 → Binary Thresholding (threshold = 155)
Step 4 → Morphological Operations (Close → Erode 14× → Dilate 13×)
Step 5 → Canny Edge Detection (auto threshold)
Step 6 → Contour Detection → Draw & Highlight tumor region
```

**Outputs**: 6 intermediate images + final annotated contour image with:
- Red contour outlines around detected regions
- Semi-transparent red overlay on the largest contour (tumor location)
- Tumor area (pixels) and tumor-to-brain ratio

---

## 6. Volume & Risk Analysis

After detection, the system performs **automated 3D volume estimation** from the 2D contour:

- Assumes MRI pixel spacing of **0.5 mm/pixel**
- Uses **equivalent-sphere approximation**: `A = πr²` → `V = (4/3)πr³`
- Classifies risk level:

| Volume | Risk Level | Clinical Recommendation |
|---|---|---|
| < 1 cm³ | 🟢 Low | Monitoring or minimally invasive surgery |
| 1–10 cm³ | 🟡 Moderate | Surgical evaluation + possible radiation |
| 10–30 cm³ | 🔴 High | Multimodal treatment (surgery + radiation + chemo) |
| > 30 cm³ | 🚨 Critical | Urgent multidisciplinary review |

---

## 7. Tumor Geometric Calculator (`/calculate` endpoint)

A **standalone calculator** that lets users manually enter tumor dimensions (H × W × L in cm):

- **Cuboid Volume**: `H × W × L` cm³
- **Ellipsoidal Volume**: `(π/6) × H × W × L` cm³ *(more clinically accurate)*
- **Mass estimate** using tissue density (default: 1.05 g/cm³)
- Risk classification based on the **longest dimension**

---

## 8. Web Application (`app.py`)

A **Flask** web server with 3 routes:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the main single-page UI |
| `/analyze` | POST | Upload MRI → classification + detection + volume analysis |
| `/calculate` | POST | Standalone tumor dimension calculator |

**Tech Stack**:
- Backend: Python / Flask
- Frontend: HTML + Vanilla CSS + JavaScript (single-page app)
- CV: OpenCV (`tumor_detector.py`)
- ML: PyTorch + TorchVision (`classifier.py`)

---

## 9. Project File Structure

```
braintumor/
├── app.py                  ← Flask web server (main entry point)
├── classifier.py           ← AI classification (ResNet18 + heuristic fallback)
├── tumor_detector.py       ← OpenCV 6-step detection pipeline
├── train_brats.py          ← Full training script (BraTS 2023 dataset)
├── train_model.py          ← Alternate training script (VGG16 / FastAI)
├── debug_train.py          ← Training debug utility
├── model/
│   └── brain_tumor_resnet18.pth  ← Trained model weights (~44 MB)
├── data_slices/
│   ├── yes/               ← 4,341 tumor PNG slices
│   └── no/                ← 4,341 no-tumor PNG slices
├── dataset_extracted/     ← BraTS 2023 NIfTI volumes (from 2.zip)
├── templates/index.html   ← Frontend UI
├── static/                ← CSS, JS, uploaded images
├── presentation.html      ← Project presentation
└── report.html            ← Project report
```

---

## 10. Summary Stats at a Glance

| Metric | Value |
|---|---|
| Dataset | BraTS 2023 GLI (219 patients) |
| Total images prepared | 8,682 |
| Images used in training | ~1,200 |
| Model | ResNet18 (Transfer Learning) |
| Training stages | 2 (frozen → fine-tuned) |
| Total epochs | 18 (10 frozen + 8 fine-tune) |
| Model size | ~44 MB |
| Detection pipeline steps | 6 (OpenCV) |
| App routes | 3 |
| Fallback mode | ✅ Heuristic (works without GPU) |
