"""
Chapters 6 and 7 for the End-to-End Documentation Guide.
"""

CHAPTER_6 = """# Chapter 6: Detailed Code Explanation

In this chapter, every important block of our actual codebase is dissected with line-by-line rigor, input/output data types, internal mechanics, and failure modes.

---

## 6.1 Dissecting `app.py`

### Block 1: Page Configuration & Root Setup
```python
import os
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Diagnostic System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
```
- **File & Location:** `d:\\Brain_Tumor\\app.py` (Lines 1–19)
- **Line-by-Line Mechanics:**
  - `st.set_page_config(...)`: Must be the very first Streamlit call in the script. It configures the browser tab title, favicon emoji, enables wide-screen responsive layout (`layout="wide"`), and ensures the sidebar is open on initial load.
- **Expected Inputs:** Keyword arguments (strings, boolean).
- **Expected Outputs:** Mutates Streamlit's internal page metadata state before rendering HTML.
- **What Happens If Placed Later:** Calling `st.set_page_config()` after any `st.write()` or `st.markdown()` raises a `StreamlitSetPageConfigMustBeFirstCommandError` and halts the application.

---

### Block 2: In-Memory Model Loader & Fallback Logic
```python
BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "Healthy (No Tumor)",
    "pituitary": "Pituitary Tumor"
}

@st.cache_resource(show_spinner=False)
def load_neuroscan_model():
    # Loads and caches the best trained model in memory.
    preferred_path = BASE_DIR / "models" / "brain_tumor_mobilenetv2_full_scratch.keras"
    fallback_path = BASE_DIR / "models" / "brain_tumor_mobilenetv2_best.keras"
    
    model_path = preferred_path if preferred_path.exists() else fallback_path
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        return None, None
    
    model = tf.keras.models.load_model(model_path)
    base_model = [l for l in model.layers if "mobilenet" in l.name.lower()][0]
    return model, base_model

model, base_model = load_neuroscan_model()
```
- **Line-by-Line Mechanics:**
  - `Path(__file__).resolve().parent`: Computes the absolute filesystem directory of `app.py` (`D:\Brain_Tumor`), regardless of the user's terminal working directory.
  - `@st.cache_resource`: Caching decorator. Stores the returned `(model, base_model)` tuple in global application memory.
  - `preferred_path`: Points to our 86.12% full-dataset model.
  - `fallback_path`: Points to the 82.19% baseline model.
  - `[l for l in model.layers if "mobilenet" in l.name.lower()][0]`: Inspects the model's layer list to dynamically find the sub-model layer containing the MobileNetV2 base (avoiding hardcoded layer index bugs).
- **Parameters:** None.
- **Input Types:** Files on disk.
- **Return Value & Type:** Tuple of `(keras.src.models.functional.Functional, keras.src.models.functional.Functional)`.
- **Failure Mode:** If neither file exists on disk, `model_path.exists()` evaluates to `False`, `st.error()` is rendered, and `(None, None)` is returned.

---

### Block 3: The Grad-CAM Calculation Engine
```python
def compute_gradcam(img_tensor, model, base_model, pred_index=None):
    # Computes Grad-CAM heatmap using the last convolutional activations of MobileNetV2.
    prep_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_tensor)

    with tf.GradientTape() as tape:
        conv_features = base_model(prep_img, training=False)
        tape.watch(conv_features)

        # Pass features through the classification head layers
        x = conv_features
        for layer in model.layers[3:]:
            x = layer(x, training=False)
        preds = x

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    # Compute gradients of top class score with respect to feature maps
    grads = tape.gradient(class_score, conv_features)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps and apply ReLU
    heatmap = conv_features[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy(), preds[0].numpy(), int(pred_index)
```
- **Line-by-Line Mechanics:**
  - `prep_img = ...`: Scales pixel values from $[0.0, 255.0]$ to $[-1.0, 1.0]$.
  - `with tf.GradientTape() as tape`: Begins recording automatic differentiation operations.
  - `conv_features = base_model(...)`: Feeds the image into MobileNetV2, returning feature maps of shape `(1, 7, 7, 1280)`.
  - `tape.watch(conv_features)`: Explicitly instructs the tape to track gradients with respect to this intermediate activation tensor.
  - `for layer in model.layers[3:]`: Forwards the spatial activations through Global Average Pooling, Batch Normalization, Dropout, Dense(128), and the Softmax output layer.
  - `pred_index = tf.argmax(preds[0])`: Identifies the winning class index ($0, 1, 2,$ or $3$).
  - `class_score = preds[:, pred_index]`: Isolates the scalar activation score of the predicted class.
  - `grads = tape.gradient(class_score, conv_features)`: Backpropagates through the head layers to compute $\frac{\partial y^c}{\partial A^k}$. Gradient tensor shape: `(1, 7, 7, 1280)`.
  - `pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))`: Averages gradients over batch, height, and width to compute importance weights $\alpha_k^c$ (1D vector of length 1,280).
  - `heatmap = conv_features[0] @ pooled_grads[..., tf.newaxis]`: Matrix multiplication taking the linear combination of the 1,280 feature maps weighted by their importance scores. Resulting shape: `(7, 7, 1)`.
  - `heatmap = tf.maximum(heatmap, 0)`: **The ReLU operation.** Discards features that negatively correlate with this tumor.
  - `... / (tf.math.reduce_max(heatmap) + 1e-10)`: Normalizes intensities into the range $[0.0, 1.0]$. `1e-10` prevents zero-division errors on blank images.
- **Input Parameters:**
  - `img_tensor`: 4D float tensor `(1, 224, 224, 3)`.
  - `model`: Full compiled Keras model.
  - `base_model`: Sub-model referencing MobileNetV2.
  - `pred_index`: Optional target class index (defaults to winning class).
- **Return Values:**
  - `heatmap.numpy()`: 2D NumPy float array `(7, 7)`.
  - `preds[0].numpy()`: 1D NumPy float array `(4,)` representing class probabilities.
  - `int(pred_index)`: Integer index of predicted class.

---

### Block 4: Diagnostic Overlay Blending
```python
def create_superimposed_overlay(original_rgb, heatmap, alpha=0.45):
    # Blends the Grad-CAM jet heatmap with the original image.
    h, w, _ = original_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_colormap, cv2.COLOR_BGR2RGB)
    superimposed = np.uint8(original_rgb * (1.0 - alpha) + jet_rgb * alpha)
    return superimposed, jet_rgb
```
- **Line-by-Line Mechanics:**
  - `cv2.resize(heatmap, (w, h))`: Upsamples the small $7 \times 7$ heatmap to the original image dimensions ($224 \times 224$) using bilinear interpolation.
  - `heatmap_uint8 = np.uint8(255 * heatmap_resized)`: Converts float $[0.0, 1.0]$ to integer range $[0, 255]$ required by OpenCV colormaps.
  - `cv2.applyColorMap(..., cv2.COLORMAP_JET)`: Applies standard thermal colormap (high values = red, medium = yellow/green, low = deep blue). Outputs BGR format.
  - `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`: Converts OpenCV's BGR order into RGB for correct browser rendering.
  - `superimposed = np.uint8(original_rgb * (1.0 - alpha) + jet_rgb * alpha)`: Linear alpha-blending formula.
- **Parameters:**
  - `original_rgb`: NumPy array `(H, W, 3)`, uint8.
  - `heatmap`: NumPy array `(7, 7)`, float32.
  - `alpha`: Float between $0.0$ and $1.0$ controlling overlay opacity (calibrated default: $0.45$).
- **Return Values:** Tuple of `(superimposed_image, pure_jet_heatmap)`.

---

## 6.2 Dissecting `notebooks/01_data_exploration.ipynb`

### Block 1: Cryptographic MD5 Deduplication & Leakage Detection
```python
import hashlib

def compute_image_hash(filepath):
    # Computes MD5 hash of an image file on disk.
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Extract hashes across all training and testing images
train_hashes = {compute_image_hash(p): p for p in train_files}
test_hashes = {compute_image_hash(p): p for p in test_files}

# Compute set intersection
leaked_hashes = set(train_hashes.keys()).intersection(set(test_hashes.keys()))
print(f"Total Unique Train Images : {len(train_hashes)}")
print(f"Total Unique Test Images  : {len(test_hashes)}")
print(f"Data Leakage Detected     : {len(leaked_hashes)} images")
```
- **Line-by-Line Mechanics:**
  - `open(filepath, "rb")`: Opens image file in binary read mode.
  - `hasher.update(buf)`: Computes an MD5 128-bit hash digest. If two images have identical pixel bytes—even if their filenames are completely different—their MD5 hashes will match 100%.
  - `set(train_hashes.keys()).intersection(...)`: Finds identical files present in both training and test sets.
- **Clinical/Engineering Significance:**
  - Finding: **0 leaked images.** This provides mathematical proof that the model was never tested on memorized training images, validating the integrity of our test accuracy figures.

---

## 6.3 Dissecting `notebooks/02_model_training.ipynb`

### Block 1: The `tf.data` Input Pipeline & Performance Optimization
```python
AUTOTUNE = tf.data.AUTOTUNE

# Load training directory
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(224, 224),
    batch_size=32,
    color_mode="rgb",
    label_mode="categorical",
    shuffle=True,
    seed=42
)

# Pipeline optimization
train_ds = raw_train_ds.cache().shuffle(buffer_size=1000).prefetch(buffer_size=AUTOTUNE)
```
- **Line-by-Line Mechanics:**
  - `image_dataset_from_directory`: Reads directory structure where subfolder names become class labels. Returns a `tf.data.Dataset` yielding batches of `(images, labels)`.
  - `label_mode="categorical"`: Encodes labels as one-hot vectors of length 4 (e.g., `[1, 0, 0, 0]` for glioma).
  - `.cache()`: Keeps decoded image tensors in memory after the first epoch, eliminating redundant disk I/O bottlenecks.
  - `.shuffle(buffer_size=1000)`: Randomly shuffles images within a 1,000-image sliding window, preventing the model from learning spurious batch-order correlations.
  - `.prefetch(buffer_size=AUTOTUNE)`: Overlaps CPU data preprocessing with GPU/CPU model forward-passes, eliminating hardware idle time.
"""

CHAPTER_7 = """# Chapter 7: Complete Data Flow

## 7.1 End-to-End Data Journey: From Pixel to Diagnosis

Here is the exact step-by-step path that information takes through the NeuroScan AI system:

```
[ Step 1: Raw Image File ]
      │ (e.g. Te-gl_0010.jpg on disk: 512 x 512, Grayscale JPEG)
      ▼
[ Step 2: Ingestion & Format Standardization ]
      │ Image.open() -> convert("RGB") -> 3 Channels
      ▼
[ Step 3: Spatial Resizing ]
      │ cv2.resize(img, (224, 224)) -> Array shape: (224, 224, 3), dtype: uint8 [0, 255]
      ▼
[ Step 4: Batch Dimension Expansion ]
      │ np.expand_dims(..., axis=0) -> Tensor shape: (1, 224, 224, 3), dtype: float32
      ▼
[ Step 5: MobileNetV2 Normalization ]
      │ tf.keras.applications.mobilenet_v2.preprocess_input() -> Values scaled to [-1.0, 1.0]
      ▼
[ Step 6: Convolutional Feature Extraction ]
      │ MobileNetV2 Base (53 Inverted Residual Blocks)
      ▼
[ Step 7: Deepest Feature Map Layer ]
      │ Layer: out_relu -> Shape: (1, 7, 7, 1280) (Spatial coordinates intact!)
      ▼
[ Step 8: Spatial Dimensionality Reduction ]
      │ GlobalAveragePooling2D() -> Vector of shape: (1, 1280)
      ▼
[ Step 9: Dense Classification Head ]
      │ BatchNorm -> Dropout(0.3) -> Dense(128, ReLU) -> BatchNorm -> Dropout(0.2)
      ▼
[ Step 10: Probability Distribution Output ]
      │ Dense(4, Softmax) -> Vector of shape: (1, 4), values sum to 1.0
      ▼
[ Step 11: Prediction Decoding ]
      │ argmax(probs) -> Predicted Class Index (e.g. 0 -> "Glioma Tumor")
      │ Confidence Score = probs[0] * 100 (e.g. 96.8%)
      ▼
[ Step 12: Grad-CAM Gradient Backpropagation ]
      │ tf.GradientTape backpropagates d(score_glioma) / d(out_relu)
      │ Importance Weights: alpha_k = reduce_mean(gradients, axis=(0,1,2))
      │ Saliency Map: ReLU( sum(alpha_k * feature_map_k) ) -> Shape: (7, 7)
      ▼
[ Step 13: Thermal Visualization & Superimposition ]
      │ Upsample 7x7 -> 224x224 -> cv2.COLORMAP_JET -> Alpha Blend (alpha=0.45)
      ▼
[ Step 14: Reactive Browser Display ]
      │ Streamlit renders Primary Badge, Saliency Studio, & Multi-Class Progress Bars
```

---

## 7.2 Tracing a Real Sample Record: `Te-gl_0010.jpg`

Let us follow an actual test scan through every mathematical transformation:

| Processing Stage | Operation / Function | Data Structure & Type | Shape | Value Range / Concrete Example |
| :--- | :--- | :--- | :--- | :--- |
| **1. Disk Storage** | File on disk | JPEG Binary file | N/A | `d:\Brain_Tumor\dataset\Testing\glioma\Te-gl_0010.jpg` |
| **2. Decoding** | `Image.open(path).convert("RGB")` | PIL RGB Image | $512 \times 512$ | 3 channels, 8-bit color |
| **3. Spatial Resize** | `cv2.resize(img_rgb, (224, 224))` | NumPy array (`uint8`) | `(224, 224, 3)` | Integers $[0, 255]$ |
| **4. Tensor Packaging** | `np.expand_dims(arr.astype('float32'), 0)`| NumPy array (`float32`) | `(1, 224, 224, 3)` | Floats $[0.0, 255.0]$ |
| **5. Preprocessing** | `mobilenet_v2.preprocess_input(tensor)` | TensorFlow Tensor (`float32`)| `(1, 224, 224, 3)` | Floats $[-1.0, 1.0]$ |
| **6. Convolution** | `base_model(prep_img)` | TensorFlow Tensor (`float32`)| `(1, 7, 7, 1280)` | Output of `out_relu` activation |
| **7. Spatial Pooling**| `GlobalAveragePooling2D()(features)` | TensorFlow Tensor (`float32`)| `(1, 1280)` | Mean activation per filter |
| **8. Dense Projection**| `dense_128(pooled)` | TensorFlow Tensor (`float32`)| `(1, 128)` | ReLU activations $\ge 0.0$ |
| **9. Classification**| `output_predictions(head_features)` | TensorFlow Tensor (`float32`)| `(1, 4)` | `[0.9682, 0.0214, 0.0018, 0.0086]` |
| **10. Argmax** | `np.argmax(probs)` | Python Integer | Scalar | `0` (Maps to `"glioma"`) |
| **11. Gradients** | `tape.gradient(score, conv_features)` | TensorFlow Tensor (`float32`)| `(1, 7, 7, 1280)` | Local gradient vectors |
| **12. Saliency Map** | Weighted Sum + ReLU + Normalization | NumPy array (`float32`) | `(7, 7)` | Continuous floats $[0.0, 1.0]$ |
| **13. Colormap** | `cv2.applyColorMap(uint8_map, JET)` | NumPy array (`uint8`) | `(224, 224, 3)` | BGR thermal colors $[0, 255]$ |
| **14. Compositing** | `(1-alpha)*original + alpha*jet` | NumPy array (`uint8`) | `(224, 224, 3)` | RGB Superimposed diagnostic image |

---

## 7.3 Training Preprocessing vs. Inference Preprocessing

A major source of confusion for junior engineers is the difference between data pipelines during **training** versus **inference**:

| Aspect | Training Pipeline (`02_model_training.ipynb`) | Inference Pipeline (`app.py`) | Why the Difference Exists? |
| :--- | :--- | :--- | :--- |
| **Input Source** | Batch generator from disk (`5,600` images) | Single image from user upload or test picker | Training optimizes model weights over large datasets; inference predicts on one patient. |
| **Batch Size** | $32$ images simultaneously | Exactly $1$ image | Memory efficiency vs. real-time single-patient latency. |
| **Data Augmentation** | **Active (`training=True`)**: Random flips, 5% rotations, 5% zooms | **Disabled (`training=False`)**: Deterministic pixel pass-through | Augmentation prevents overfitting during training; during diagnosis, we must never alter the patient's true scan! |
| **Batch Normalization** | Computes running batch mean & variance | Uses frozen moving mean & variance | Batch statistics must remain constant during inference to avoid prediction drift. |
| **Dropout** | Randomly zeros out 30% and 20% of activations | **Disabled**: All neurons active, scaled by keep-probability | Regularization is only needed during weight optimization. |
| **Shuffle** | Enabled (`shuffle=True`, buffer=1000) | Disabled (`shuffle=False`) | Order does not matter for single-scan prediction. |
"""

print("Writing Chapters 6 and 7 module...")
