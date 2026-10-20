"""
Chapters 10, 11, and 12 for the End-to-End Documentation Guide.
"""

CHAPTER_10 = """# Chapter 10: Frontend and Backend Connection

## 10.1 The Streamlit Execution Model: In-Process Reactive Computation

A critical concept for software engineering interviews is explaining **how Streamlit actually works** compared to traditional web development:

### Traditional Web Application (Decoupled Client-Server)
```
[ User Browser (React/Vue/HTML) ]
             │
             │ HTTP POST /api/predict (JSON payload or Multipart Form)
             ▼
[ Web Server (FastAPI / Flask / Django) ]
             │
             │ Preprocesses bytes -> Model.predict() -> Formats JSON response
             ▼
[ User Browser (React) parses JSON & updates virtual DOM ]
```

### NeuroScan AI Architecture (Streamlit Reactive In-Process Model)
In our project, there is **no separate REST API, no JSON serialization layer, and no Flask/FastAPI server**.

Instead, **the frontend and backend live within the exact same Python process**:
```
+-------------------------------------------------------------------------+
|                  SINGLE STREAMLIT PYTHON RUNTIME PROCESS                |
|                                                                         |
|  1. Browser Interaction (User clicks dropdown / uploads image)          |
|                           │                                             |
|                           ▼ (Websocket Event)                           |
|  2. Streamlit re-executes app.py from Top to Bottom                     |
|                           │                                             |
|  3. Hits @st.cache_resource -> Model already in RAM! (0 ms overhead)   |
|                           │                                             |
|  4. Directly calls Python functions:                                    |
|     • compute_gradcam(img_tensor, model, base_model)                    |
|     • create_superimposed_overlay(img_resized, heatmap)                 |
|                           │                                             |
|  5. Computes NumPy arrays & updates UI elements in real time            |
|                           │                                             |
|                           ▼ (Serialized Protobuf Delta Stream)          |
|  6. Browser updates DOM elements immediately                            |
+-------------------------------------------------------------------------+
```

---

## 10.2 Sequence Diagram of User Interaction

```
User                  Browser (DOM)             Streamlit Engine              Keras Model / Grad-CAM
 │                         │                           │                                │
 ├── Selects Sample Scan ─>│                           │                                │
 │                         ├── Send WebSocket Event ──>│                                │
 │                         │   (Widget state updated)  ├── Re-executes app.py           │
 │                         │                           │                                │
 │                         │                           ├── Check @st.cache_resource ────┤
 │                         │                           │   (Returns cached model)       │
 │                         │                           │                                │
 │                         │                           ├── Preprocess image array       │
 │                         │                           │   (224x224x3, [-1, 1])         │
 │                         │                           │                                │
 │                         │                           ├── compute_gradcam() ──────────>│
 │                         │                           │   Forward pass & Backprop Tape │
 │                         │                           │<── Returns probs & 7x7 heatmap─┤
 │                         │                           │                                │
 │                         │                           ├── OpenCV overlay blending      │
 │                         │                           │   (Resize, Colormap, Alpha)    │
 │                         │                           │                                │
 │                         │<── Stream Render Deltas ──┤                                │
 │<── Sees Diagnosis & ────┤    (Cards, Images, Bars)  │                                │
 │    Glowing Grad-CAM     │                           │                                │
```

---

## 10.3 Why This Architecture Was Chosen

1. **Zero Serialization Latency:** Traditional web apps must encode image tensors into base64 strings or write temporary files to disk, transmit them over HTTP, decode them in Flask, and encode the response back into JSON. In Streamlit, tensors are passed directly through Python memory pointers in nanoseconds.
2. **Elimination of State Desynchronization:** Because the script runs top-to-bottom on state changes, there are no "stale state" bugs between frontend state and backend databases.
3. **Speed of Development:** Allows building a complete, high-performance medical AI prototype in a single, auditable Python module (`app.py`).
"""

CHAPTER_11 = """# Chapter 11: Configuration, Dependencies, and Logging

## 11.1 Dependency Specification (`requirements.txt`)

Every library in `d:\Brain_Tumor\requirements.txt` has a specific technical rationale:

```text
# Deep Learning framework and neural network API
tensorflow>=2.15.0

# Numerical computing (pinned below 2.0 for TensorFlow C-API stability)
numpy>=1.24.0,<2.0.0

# Tabular data manipulation and metrics storage
pandas>=2.0.0

# Plotting and image visualization
matplotlib>=3.7.0

# Statistical visualizations (used for heatmaps and confusion matrices)
seaborn>=0.12.0

# Machine learning utilities and classification metrics
scikit-learn>=1.3.0

# Image loading, verification, and resizing
pillow>=9.5.0

# Computer vision processing
opencv-python>=4.8.0

# Jupyter notebook kernel support inside VS Code
ipykernel>=6.25.0

# Web application framework for our interactive UI
streamlit>=1.30.0
```

---

## 11.2 Key Configuration Constants & Parameters

Throughout the codebase, specific global constants govern system behavior:

| Constant Name | Defined In | Value | Technical Role | What Breaks If Changed? |
| :--- | :--- | :--- | :--- | :--- |
| `IMG_HEIGHT`, `IMG_WIDTH` | `app.py`, Notebooks | $224, 224$ | Defines the spatial input dimension required by the MobileNetV2 architecture. | Altering to $256$ causes a dimension mismatch error in the pre-trained first convolution layer. |
| `BATCH_SIZE` | `02_model_training.ipynb` | $32$ | Number of image tensors processed simultaneously during training step. | Setting too high (e.g. 256) causes Out-Of-Memory (OOM) CPU crashes. |
| `seed` | `02_model_training.ipynb` | $42$ | Seed for TensorFlow, NumPy, and Python pseudo-random number generators. | Changing this creates different random train/validation splits, altering reproducibility. |
| `overlay_alpha` | `app.py` | $0.45$ | Transparency weight for superimposing the Grad-CAM thermal map onto the MRI. | If set to $1.0$, the original MRI is hidden; if set to $0.0$, the heatmap disappears. |
| `CLASS_NAMES` | `app.py`, Notebooks | `["glioma", "meningioma", "notumor", "pituitary"]` | Canonical list of class labels in strict alphabetical order (matching Keras directory reader). | Scrambling the list order assigns model probability outputs to the wrong clinical diagnoses! |

---

## 11.3 Logging Levels and Error Handling Best Practices

In our current implementation, user-facing error handling is managed via Streamlit UI guards:
- `st.error()`: Highlights critical failures (e.g. missing `.keras` model file).
- `st.warning()`: Alerts non-fatal issues (e.g. missing sample folder).
- `st.info()`: Instructs the user on initial state.

### Production Logging Recommendations (For Enterprise Deployment)
For a hospital-grade production environment, Python's built-in `logging` module should be introduced to create immutable audit trails:

| Level | Severity | Example in NeuroScan AI | Purpose |
| :--- | :--- | :--- | :--- |
| `DEBUG` | Lowest | `DEBUG: Image resized from (512, 512) to (224, 224) in 1.2ms` | Detailed diagnostic information for developers. |
| `INFO` | Routine | `INFO: Model brain_tumor_mobilenetv2_full_scratch.keras loaded successfully` | Confirms normal system operations and startup milestones. |
| `WARNING`| Non-fatal | `WARNING: Primary model checkpoint missing; falling back to baseline` | Flags unexpected conditions that did not halt execution. |
| `ERROR` | Severe | `ERROR: Input image decoding failed: Invalid JPEG byte header` | Logs a failed operation on a specific patient request. |
| `CRITICAL`| Catastrophic | `CRITICAL: GPU memory exhausted; process terminating` | System-wide failure requiring immediate administrative intervention. |
"""

CHAPTER_12 = """# Chapter 12: Complete End-to-End Execution Example

Let us trace a realistic walkthrough of a user operating the live system:

---

## 12.1 The Concrete Scenario: Diagnosing a Meningioma

### Phase 1: Launching the System
1. The developer opens the terminal and executes:
   ```bash
   d:\Brain_Tumor\.venv\Scripts\streamlit run app.py
   ```
2. Streamlit starts Uvicorn and launches a local server on port `8501`.
3. The browser connects to `http://localhost:8501`.
4. `app.py` runs its first pass:
   - `@st.cache_resource` loads `brain_tumor_mobilenetv2_full_scratch.keras` into memory.
   - Initial state: no image selected.
   - UI renders: Header Banner + Blue Info Card: *"Please select a pre-loaded clinical sample or upload an MRI scan from the sidebar to begin analysis."*

---

### Phase 2: User Selects a Sample Scan
1. In the sidebar, the user selects: **"🔬 Select Clinical Test Sample"**.
2. From the dropdown, the user picks: **`Meningioma Tumor (Sample #1)`**.
3. Streamlit re-executes `app.py` with the new widget state:
   - `selected_image = Image.open("dataset/Testing/meningioma/Te-me_0010.jpg")`.
   - `actual_label = "meningioma"`.

---

### Phase 3: Preprocessing & Neural Computation
1. `selected_image.convert("RGB")` converts the image to 3 channels.
2. `cv2.resize(..., (224, 224))` rescales the image to exact model dimensions.
3. `np.expand_dims(..., 0)` produces tensor of shape `(1, 224, 224, 3)`.
4. `compute_gradcam()` executes:
   - Pixel scaling: $[-1.0, 1.0]$.
   - Forward pass through MobileNetV2 base: outputs `(1, 7, 7, 1280)`.
   - Head forward pass: outputs softmax probabilities:
     $$\text{probs} = [0.0381, 0.9412, 0.0012, 0.0195]$$
   - Predicted class index: `pred_idx = 1` (`"meningioma"`).
   - Confidence score: $94.12\%$.
   - `tf.GradientTape` extracts gradients $\frac{\partial (0.9412)}{\partial (\text{out\_relu})}$.
   - Channel weights $\alpha_k$ calculated; linear combination computed; ReLU applied.
   - Returns $7 \times 7$ normalized heatmap array.

---

### Phase 4: Saliency Compositing
1. `create_superimposed_overlay()` upsamples the $7 \times 7$ heatmap to $224 \times 224$.
2. Applies OpenCV `COLORMAP_JET`: high activation pixels along the dura mater turn fiery red/yellow; brain tissue turns blue.
3. Alpha blends at $\alpha = 0.45$: creates the final diagnostic overlay.

---

### Phase 5: Browser Rendering
Within 350 milliseconds of selecting the dropdown, the browser displays:
1. **Primary Diagnosis Banner (Red Theme):**
   ```text
   ⚠️ PRIMARY DIAGNOSIS RESULT: Meningioma Tumor
   Confidence: 94.1%
   ```
2. **Left Column (Visual Saliency Studio):**
   - Shows the original brain MRI scan with a bright red glowing hotspot centered on the dural lesion.
3. **Right Column (Multi-Class Probabilities):**
   - Meningioma: `94.12%` (Long orange bar)
   - Glioma: `3.81%` (Small red bar)
   - Pituitary: `1.95%` (Small purple bar)
   - Healthy (No Tumor): `0.12%` (Tiny green bar)
4. **Validation Card:**
   ```text
   Validation: Ground Truth: meningioma | Prediction: meningioma [MATCHED]
   ```

---

## 12.2 What Happens If the Input Changes to a Healthy Brain?

If the user switches the dropdown to **`Healthy (No Tumor) (Sample #1)`**:
1. The model outputs:
   $$\text{probs} = [0.0021, 0.0018, 0.9942, 0.0019]$$
2. `is_healthy = True`.
3. The Primary Diagnosis Banner shifts dynamically from Red to **Emerald Green**:
   ```text
   ✅ PRIMARY DIAGNOSIS RESULT: Healthy (No Tumor)
   Confidence: 99.4%
   ```
4. In the Saliency Studio, the Grad-CAM heatmap shows no intense focal hotspots over brain tissue, visually verifying the absence of pathological mass effect!
"""

print("Writing Chapters 10, 11, and 12 module...")
