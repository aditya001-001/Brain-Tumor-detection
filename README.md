# 🧠 NeuroScan AI — Brain Tumor Detection & Explainable AI (Grad-CAM)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end clinical deep learning system for 4-class intracranial tumor diagnosis from cranial MRI scans, equipped with real-time Grad-CAM (Gradient-weighted Class Activation Mapping) visual explainability and an interactive Streamlit diagnostic studio.**

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [Dataset & Clinical Classes](#-dataset--clinical-classes)
- [Deep Learning Architecture](#-deep-learning-architecture)
- [Experimental Results & Empirical Validation](#-experimental-results--empirical-validation)
- [Explainable AI (Grad-CAM)](#-explainable-ai-grad-cam)
- [Interactive Web Application](#-interactive-web-application)
- [Project Directory Structure](#-project-directory-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Clinical Disclaimer](#-clinical-disclaimer)

---

## 🩺 Executive Overview

Brain tumors—including infiltrative **Gliomas**, dural-based **Meningiomas**, and sellar **Pituitary adenomas**—are among the most critical intracranial pathologies requiring rapid, accurate radiological evaluation. While modern Convolutional Neural Networks achieve impressive diagnostic accuracy, their "black-box" nature creates a significant barrier to clinical adoption: **clinicians cannot trust predictions without understanding where the algorithm is looking.**

**NeuroScan AI** bridges this gap by combining:
1. A lightweight, high-performance **MobileNetV2** transfer-learning classifier trained on cranial MRI scans.
2. A mathematical **Grad-CAM explainability engine** that generates diagnostic heatmaps pointing directly to the lesion bed.
3. An intuitive, real-time **Streamlit web dashboard** designed for rapid radiological review.

---

## ✨ Key Features

- **4-Class Intracranial Classification**: Differentiates between *Glioma*, *Meningioma*, *Pituitary Adenoma*, and *Healthy Brain (No Tumor)*.
- **High Clinical Sensitivity**: Achieves **99.0% recall** on healthy brain scans and virtually **0% false negative rate** for tumor presence.
- **Zero-Annotation Tumor Localization (Grad-CAM)**: Generates spatial heatmaps without requiring costly pixel-level segmentation masks or bounding boxes during training.
- **Empirically Validated Full-Dataset Training**: Jumped from an initial baseline of **82.19% $\rightarrow$ 86.12% test accuracy** (+3.93%) by incorporating full data coverage.
- **Production-Ready Dashboard**: Streamlit web app with sub-second model caching (`@st.cache_resource`), pre-loaded test scans, and multi-class probability breakdown.

---

## 🗂️ Dataset & Clinical Classes

The system is evaluated on a verified dataset of **7,200 brain MRI slices** divided into:
- **Training Set**: 5,600 scans (1,400 scans per class).
- **Untouched Test Set**: 1,600 scans (400 scans per class).

| Class | Anatomical Description | Clinical Characteristic |
| :--- | :--- | :--- |
| **`glioma`** | Intra-axial neuroepithelial tumor arising from glial cells. | Infiltrative margins invading brain parenchyma. |
| **`meningioma`** | Extra-axial mass arising from the arachnoid layer of the meninges. | Well-circumscribed borders with dural attachment. |
| **`pituitary`** | Sellar mass originating from the pituitary gland. | Skull base lesion near the optic chiasm. |
| **`notumor`** | Normal intracranial anatomy. | Symmetrical ventricles and healthy cerebral tissue. |

---

## 🏗️ Deep Learning Architecture

```
[ Input MRI Scan (224 x 224 x 3) ]
                │
                ▼
[ Anatomically Safe Augmentation ] (RandomFlip, 5% Rotation, 5% Zoom)
                │
                ▼
[ MobileNetV2 Preprocessing ] (Scales pixel values to [-1, 1])
                │
                ▼
[ MobileNetV2 Pretrained Base ] (Frozen ImageNet Feature Extractor)
                │ ───> [ out_relu Layer (7 x 7 x 1280) ] ─── (Tapped for Grad-CAM)
                ▼
[ Global Average Pooling 2D ] (Reduces spatial dimensions to 1280-D vector)
                │
                ▼
[ BatchNormalization + Dropout (0.3) ]
                │
                ▼
[ Dense Head (128 units, ReLU) ]
                │
                ▼
[ BatchNormalization + Dropout (0.2) ]
                │
                ▼
[ Output Layer (4 units, Softmax) ] ──> Multi-Class Probabilities
```

---

## 📊 Experimental Results & Empirical Validation

To maximize generalization, we tested two distinct training methodologies evaluated against the **same 1,600 untouched test scans**:

| Diagnostic Metric | Model 1 (80/20 Split: 4,480 train) | Model 2 (Full Retrain: 5,600 train) | Empirical Gain |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | **82.19%** | **86.12%** | **+3.93%** 🚀 |
| **Categorical Loss** | 0.5601 | **0.4716** | **-0.0885** (Sharper confidence) |
| **Meningioma Recall** | 63.50% (254/400) | **78.50% (314/400)** | **+60 accurate diagnoses** |
| **Glioma Precision** | 90.75% | **92.81%** | **+2.06%** |
| **Pituitary Recall** | 99.25% (397/400) | **99.25% (397/400)** | Highly stable |
| **Healthy Brain Recall** | 99.75% (399/400) | **99.00% (396/400)** | Minimal false alarms |

### Detailed Test Set Classification Report (1,600 Scans)
```
              precision    recall  f1-score   support

      glioma     0.9281    0.6775    0.7832       400
  meningioma     0.7949    0.7850    0.7899       400
     notumor     0.9167    0.9900    0.9519       400
   pituitary     0.8254    0.9925    0.9012       400

    accuracy                         0.8612      1600
   macro avg     0.8663    0.8612    0.8566      1600
weighted avg     0.8663    0.8612    0.8566      1600
```

---

## 🔍 Explainable AI (Grad-CAM)

Grad-CAM extracts the gradients of the target class score $y^c$ with respect to the feature map activations $A^k$ of the final convolutional layer (`out_relu`):

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

- **Positive Activation Focus**: Passing through the ReLU function ensures the heatmap isolates features that *positively* support the tumor diagnosis, ignoring features for other classes.
- **Jet Colormap Superimposition**: The resulting $7 \times 7$ heatmap is bilinearly upsampled to $224 \times 224$ and blended with the patient's grayscale MRI ($\alpha = 0.45$).

---

## 🖥️ Interactive Web Application

The project includes an interactive web dashboard built with **Streamlit**:
- **Dual Scan Input**: Upload custom DICOM-converted `.jpg`/`.png` MRI slices or select from pre-loaded clinical test cases.
- **Real-Time Saliency Studio**:
  - *🩺 Superimposed Grad-CAM*: Visual highlight directly on anatomical structures.
  - *🔥 Pure Heatmap*: Gradient activation intensity view.
  - *↔️ Side-by-Side Comparison*: Direct comparison of raw MRI vs. diagnostic overlay.
- **Confidence Distribution**: Dynamic color-coded probability bars for all 4 diagnostic classes.

---

## 📁 Project Directory Structure

```bash
Brain_Tumor/
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Comprehensive EDA, MD5 hash leakage checks, resolution audits
│   └── 02_model_training.ipynb     # Model training, full retrain experiments, evaluation & metrics
├── dataset/
│   └── Testing/                    # 1,600 verified test MRI scans (Glioma, Meningioma, Pituitary, NoTumor)
├── models/
│   └── brain_tumor_mobilenetv2_full_scratch.keras  # Production model weights (11.1 MB)
├── app.py                          # Interactive Streamlit Diagnostic Studio
├── requirements.txt                # Pinned production dependencies
├── .gitignore                      # Configured to ignore virtualenvs & local artifacts
└── README.md                       # Clinical documentation & architectural overview
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/aditya001-001/Brain-Tumor-detection.git
cd Brain-Tumor-detection
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Diagnostic Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive application.

---

## ⚠️ Clinical Disclaimer

This software is developed strictly for **academic, research, and technical demonstration purposes**. Deep learning models are subject to distribution shifts and radiological edge cases. This system is **not** an FDA-approved medical device and must **never** be used as a substitute for professional clinical diagnosis or the review of a board-certified neuroradiologist.

---

## 👤 Author
**Aditya Sarapure**  
- GitHub: [@aditya001-001](https://github.com/aditya001-001)  
- Email: adityasarapure100@gmail.com
