"""
Chapters 13 and 14 for the End-to-End Documentation Guide.
"""

CHAPTER_13 = """# Chapter 13: Testing and Troubleshooting

## 13.1 Existing Testing Mechanisms in the Codebase

In our project, testing was implemented through empirical validation and scientific auditing routines:

1. **Cryptographic Data Leakage Test (`01_data_exploration.ipynb`):**
   - Verified that no identical MRI slices exist in both the Training and Testing splits by comparing MD5 hashes across all 7,200 images.
   - Result: **0 leaked images (100% clean split separation).**
2. **Channel & Dimension Integrity Test (`01_data_exploration.ipynb`):**
   - Scanned all images to identify unique resolutions (447 unique sizes) and color modes (RGB, Grayscale L, RGBA), establishing the strict requirement for 3-channel conversion and $224 \times 224$ resizing.
3. **Out-of-Sample Generalization Test (`02_model_training.ipynb`):**
   - Benchmarked performance against 1,600 untouched test scans using Scikit-Learn's `classification_report` and `confusion_matrix`.
4. **Model Deserialization & Syntax Test:**
   - Validated that `brain_tumor_mobilenetv2_full_scratch.keras` loads cleanly in Python and compiles without syntax warnings via `py_compile app.py`.

---

## 13.2 Recommended Automated Unit Test Suite (`tests/test_pipeline.py`)

For enterprise continuous integration (CI/CD), we recommend formalizing these unit tests using `pytest`:

```python
import pytest
import numpy as np
import tensorflow as tf
from app import load_neuroscan_model, compute_gradcam, create_superimposed_overlay

@pytest.fixture(scope="session")
def loaded_models():
    # Loads the model once for all test cases.
    model, base_model = load_neuroscan_model()
    return model, base_model

def test_model_loading(loaded_models):
    # Verify that both outer model and base model are successfully instantiated.
    model, base_model = loaded_models
    assert model is not None, "Outer model failed to load from disk"
    assert base_model is not None, "Base MobileNetV2 model layer could not be located"

def test_prediction_output_shape_and_probability(loaded_models):
    # Ensure dummy input produces a valid probability vector of length 4 summing to 1.0.
    model, _ = loaded_models
    dummy_input = np.random.uniform(0, 255, (1, 224, 224, 3)).astype("float32")
    preds = model.predict(dummy_input)
    
    assert preds.shape == (1, 4), f"Expected shape (1, 4), got {preds.shape}"
    assert np.isclose(np.sum(preds[0]), 1.0, atol=1e-5), "Probabilities do not sum to 1.0"
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0), "Probabilities out of [0, 1] range"

def test_gradcam_output_shape_and_normalization(loaded_models):
    # Verify Grad-CAM produces a 7x7 heatmap with values normalized strictly in [0.0, 1.0].
    model, base_model = loaded_models
    dummy_input = np.random.uniform(0, 255, (1, 224, 224, 3)).astype("float32")
    heatmap, probs, pred_idx = compute_gradcam(dummy_input, model, base_model)
    
    assert heatmap.shape == (7, 7), f"Expected heatmap shape (7, 7), got {heatmap.shape}"
    assert heatmap.min() >= 0.0, "Heatmap contains negative values (ReLU failed)"
    assert heatmap.max() <= 1.0, "Heatmap max value exceeds 1.0"
    assert pred_idx in [0, 1, 2, 3], f"Invalid class index {pred_idx}"

def test_superimposed_overlay_shape():
    # Verify overlay returns a 3-channel RGB image matching original dimensions.
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_heatmap = np.ones((7, 7), dtype=np.float32)
    overlay, jet_map = create_superimposed_overlay(dummy_image, dummy_heatmap, alpha=0.45)
    
    assert overlay.shape == (224, 224, 3), "Overlay dimension mismatch"
    assert overlay.dtype == np.uint8, "Overlay must be 8-bit unsigned integer"
```

---

## 13.3 Comprehensive Troubleshooting Matrix

| Issue & Error Message | Root Cause | Verification Step | Fix / Resolution |
| :--- | :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'tensorflow'` | Python command was executed using global system Python rather than the project virtual environment. | Run `which python` or `Get-Command python` to check the executable path. | Execute using the virtual environment: `d:\Brain_Tumor\.venv\Scripts\python.exe` or run `.\.venv\Scripts\activate` first. |
| `NameError: name 'model' is not defined` | The Jupyter notebook kernel was restarted or reloaded, wiping variable state from RAM. | Look at the cell execution counter `In [1]`. | Re-run setup cells (Cells 0, 1, 4) and reload the model checkpoint via `tf.keras.models.load_model()`. |
| `Model file not found at: models/...` | The `.keras` file was moved, renamed, or training was not completed. | Run `Test-Path models\brain_tumor_mobilenetv2_full_scratch.keras`. | Re-run Cell 14 in `02_model_training.ipynb` to save the checkpoint or restore from backup. |
| `Port 8501 is already in use` | A previous instance of Streamlit is still running in the background on port 8501. | Run `netstat -ano \| findstr :8501` to find the process ID (PID). | Kill the process via `Stop-Process -Id <PID> -Force` or run on a new port: `streamlit run app.py --server.port 8502`. |
| `ValueError: Input 0 of layer is incompatible: expected shape=(None, 224, 224, 3)` | An image with arbitrary dimensions was passed without resizing, or a grayscale image (1 channel) was passed. | Print `img.shape` before feeding to model. | Ensure `img.convert('RGB')` and `cv2.resize(img, (224, 224))` run before tensor packaging. |
| `ImportError: numpy.core.multiarray failed to import` | NumPy 2.0 was installed in an environment with a TensorFlow version built for NumPy 1.x. | Run `python -c "import numpy; print(numpy.__version__)"`. | Downgrade NumPy: `pip install "numpy>=1.24.0,<2.0.0"`. |
| `UnicodeEncodeError: 'charmap' codec can't encode characters` | Windows command prompt terminal attempting to print UTF-8 emojis or progress bars under CP1252 codepage. | Check terminal encoding via `chcp`. | Set terminal codepage to UTF-8 via `chcp 65001` or add `sys.stdout.reconfigure(encoding='utf-8')`. |
| `ResourceExhaustedError: OOM when allocating tensor` | GPU or CPU RAM exhausted during batch generation or model training. | Check memory usage in Task Manager. | Reduce `BATCH_SIZE` from 32 to 16, or close memory-heavy applications. |
| Streamlit web page displays a blank screen or loops indefinitely | Browser cache holds stale JavaScript chunks from a previous Streamlit version. | Open Developer Tools (F12) -> Console. | Perform a hard refresh: `Ctrl + F5` or open in an Incognito / Private window. |
| `IndexError: list index out of range` in `base_model` lookup | Layer name searching for `"mobilenet"` did not match any layer in the outer model. | Print `[l.name for l in model.layers]` to inspect actual layer names. | Check layer naming convention and ensure model was instantiated from MobileNetV2. |
"""

CHAPTER_14 = """# Chapter 14: Project Interview Preparation

## 14.1 Elevator Pitches for Interviews

### The 30-Second Elevator Pitch
> *"I built **NeuroScan AI**, an end-to-end medical deep learning system for four-class brain tumor diagnosis from MRI scans. Using transfer learning on MobileNetV2, I trained a model that achieved **86.12% test accuracy** across 1,600 unseen scans, with **99.0% recall** on healthy brain scans. Because clinical AI cannot be a black box, I implemented a mathematical **Grad-CAM engine** that backpropagates gradients to highlight the exact tumor site in real time, serving the entire system through an interactive Streamlit diagnostic dashboard."*

### The 2-Minute Project Walkthrough
> *"In clinical radiology, diagnosing intracranial tumors—gliomas, meningiomas, and pituitary adenomas—is time-critical, yet traditional AI models face deep skepticism because they are opaque black boxes. 
> 
> To solve this, I designed **NeuroScan AI**. I began with an exploratory data audit of 7,200 cranial MRI scans, using MD5 cryptographic hashing to mathematically prove zero data leakage between training and testing splits. I built an optimized `tf.data` pipeline incorporating anatomically safe augmentations like subtle rotations and horizontal flips.
> 
> For the model, I leveraged MobileNetV2 pre-trained on ImageNet. Its depthwise separable convolutions reduced computational complexity by roughly 8x compared to standard CNNs, allowing sub-second CPU inference. I froze the base feature extractor and designed a custom classification head using Global Average Pooling, Batch Normalization, and Dropout. 
> 
> During development, I tested an engineering hypothesis: retraining on 100% of the training data (5,600 images) boosted test accuracy from an 82.19% baseline to **86.12%**, surging meningioma sensitivity from 63.5% to 78.5%. 
> 
> To provide clinical transparency, I implemented a custom Grad-CAM module using TensorFlow's `GradientTape` to compute gradients of the winning class score with respect to the final convolutional feature maps, creating an alpha-blended thermal overlay. Finally, I deployed the application via a Streamlit dashboard with in-memory model caching, providing doctors with instant classifications, probability distributions, and visual saliency maps."*

---

## 14.2 Forty Project-Specific Interview Questions and Answers

### Beginner Questions (1–10)

#### 1. What does NeuroScan AI do?
It classifies cranial MRI scans into four categories (Glioma, Meningioma, Pituitary tumor, or Healthy Brain) and highlights the physical location of the tumor using a Grad-CAM thermal heatmap.

#### 2. What programming language and deep learning library did you use?
Python 3.11 with TensorFlow and Keras, alongside OpenCV, NumPy, Scikit-learn, and Streamlit.

#### 3. What are the four classes in your dataset?
`glioma` (intra-axial glial tumor), `meningioma` (extra-axial dural tumor), `pituitary` (sellar adenoma), and `notumor` (normal healthy brain).

#### 4. What is an MRI?
Magnetic Resonance Imaging: a medical imaging modality using strong magnetic fields and radio waves to produce high-contrast cross-sectional images of soft tissue like the brain.

#### 5. What input size does your model expect?
$224 \times 224$ pixels with 3 color channels (RGB).

#### 6. Why do we resize images to 224x224?
The pre-trained MobileNetV2 architecture expects fixed dimensions ($224 \times 224$). Our raw dataset contained 447 unique resolutions; standardizing them allows batch tensor processing.

#### 7. What is transfer learning?
Taking a neural network pre-trained on a massive generic dataset (ImageNet with 1.4 million images) and repurposing its learned feature extractors for a specialized target domain (brain tumor MRIs).

#### 8. What is the difference between training data and testing data?
Training data (5,600 images) is used by the optimizer to adjust model weights. Testing data (1,600 images) is kept completely untouched to measure real-world performance on unseen data.

#### 9. What is accuracy?
The ratio of correct predictions to total predictions: $\frac{\text{Correct}}{\text{Total}}$. Our model achieved 86.12% overall test accuracy.

#### 10. Why is accuracy alone not enough in medical imaging?
Because missing a tumor (a False Negative) is life-threatening, while a false alarm (False Positive) merely triggers a follow-up scan. We must examine Recall and Confusion Matrices.

---

### Intermediate Questions (11–25)

#### 11. What is MobileNetV2?
An efficient convolutional neural network architecture designed by Google that uses Inverted Residual blocks with linear bottlenecks and Depthwise Separable Convolutions to drastically reduce computational operations.

#### 12. How does Depthwise Separable Convolution work?
It splits standard convolution into:
1. **Depthwise convolution:** Applies a single spatial filter per channel.
2. **Pointwise convolution:** Applies a $1 \times 1$ convolution to mix channels. This reduces computation by roughly 8 to 9 times.

#### 13. Why did you freeze the base model during initial training?
To prevent **Catastrophic Forgetting**. The randomly initialized dense head generates chaotic gradients in early epochs that would destroy pre-trained ImageNet filters if allowed to backpropagate into the base.

#### 14. What is Global Average Pooling 2D (GAP)?
It averages all spatial pixels in each feature map: a $(7 \times 7 \times 1280)$ tensor becomes a flat vector of $(1280)$. It eliminates millions of dense weights compared to a Flatten layer, preventing overfitting.

#### 15. Why did you include Batch Normalization?
It normalizes neuron activations by subtracting batch mean and dividing by batch standard deviation. This stabilizes training, mitigates internal covariate shift, and acts as a mild regularizer.

#### 16. What is Dropout, and what rates did you use?
Dropout randomly sets a fraction of neuron activations to zero during training to prevent co-adaptation. We used 30% (`0.3`) after GAP and 20% (`0.2`) after the Dense(128) layer.

#### 17. What loss function did you use, and why?
**Categorical Cross-Entropy**. Our problem is multi-class classification where each MRI belongs to exactly one of four mutually exclusive classes.

#### 18. What optimizer was chosen?
**Adam** with an initial learning rate of $0.001$. Adam combines the benefits of AdaGrad (adaptive learning rates) and RMSProp (moving averages of squared gradients).

#### 19. What data augmentations did you apply, and why were they restricted to 5%?
Horizontal flips, 5% rotation (~18 degrees), and 5% zoom. We restricted rotation and zoom because cranial MRI scans are naturally aligned in a scanner head coil; extreme tilts or crops violate clinical plausibility.

#### 20. What is data leakage, and how did you verify your dataset was free of it?
Data leakage occurs when test images are accidentally present in the training set, causing inflated test scores. We computed MD5 cryptographic hashes for all 7,200 images and confirmed **zero hash intersections** between train and test splits.

#### 21. What is `tf.data.AUTOTUNE`?
A TensorFlow setting that dynamically tunes buffer sizes at runtime to balance CPU prefetching and GPU/CPU consumption, preventing hardware stalls.

#### 22. What was the empirical benefit of retraining on the full dataset?
Test accuracy on the same 1,600 untouched scans jumped from **82.19% to 86.12%** (+3.93%), and **Meningioma recall surged from 63.5% to 78.5%** (+60 correctly identified tumor patients).

#### 23. What is the test recall for healthy brains (`notumor`)?
**99.0%** (396 out of 400 healthy scans correctly recognized, with near-zero false alarms).

#### 24. What is the test recall for pituitary tumors?
**99.25%** (397 out of 400 correctly classified). Sellar region tumors have distinct anatomical locations easily captured by MobileNetV2.

#### 25. Which two classes exhibit the highest confusion?
Glioma and Meningioma. 77 test gliomas were classified as meningiomas because both can appear as contrast-enhancing mass lesions when the base feature extractor is frozen.

---

### Advanced Questions (26–35)

#### 26. What is Grad-CAM, and how does it work mathematically?
Grad-CAM computes the gradient of the winning class score $y^c$ with respect to the feature map activations $A^k$ of the last conv layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
It then computes a weighted sum of the feature maps and applies a ReLU:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

#### 27. Why must we tap the **last** convolutional layer for Grad-CAM?
Because earlier conv layers only know about simple edges without semantic meaning, while dense layers have flattened away all spatial 2D coordinates. The last conv layer (`out_relu`, $7 \times 7$) is the exact boundary where semantic understanding and 2D spatial coordinates coexist.

#### 28. Why does Grad-CAM apply a ReLU at the end?
Because we only care about features that have a **positive influence** on predicting the target tumor class. Features with negative gradients indicate evidence for *competing* classes.

#### 29. How does `tf.GradientTape` function in `compute_gradcam`?
It records forward-pass tensor operations inside its Python context. Calling `tape.gradient(target, sources)` traverses the recorded computational graph backward using reverse-mode automatic differentiation.

#### 30. How is the 7x7 Grad-CAM heatmap mapped onto a 224x224 MRI?
It is bilinearly upsampled to $224 \times 224$, scaled to $[0, 255]$ uint8, converted to RGB thermal colors via OpenCV's `COLORMAP_JET`, and blended with the original MRI:
$$\text{Overlay} = (1 - \alpha) \cdot \text{Scan} + \alpha \cdot \text{JetMap}$$

#### 31. What is the purpose of `@st.cache_resource` in `app.py`?
It caches the loaded TensorFlow Keras model in memory across user sessions, preventing a costly 12 MB disk deserialization on every UI button click or slider interaction.

#### 32. In what format is your model saved on disk?
Native **Keras v3 (`.keras`)**, which is a ZIP archive containing a `config.json` model topology, HDF5 weight matrices, and metadata.

#### 33. Why shouldn't Python's `pickle` be used to save deep learning models?
Pickle files allow arbitrary code execution during deserialization via the `__reduce__()` exploit. Loading an untrusted `model.pkl` allows malicious shell command execution on hospital servers.

#### 34. How does Streamlit's architecture differ from a Flask + React setup?
Streamlit runs frontend rendering and backend computation in the **exact same Python process**. State changes trigger top-to-bottom re-execution of `app.py`, updating the DOM via WebSocket deltas without REST API endpoints or JSON overhead.

#### 35. What is the difference between Fine-Tuning and Feature Extraction?
- **Feature Extraction:** Base model is frozen (`trainable = False`); only custom head weights update.
- **Fine-Tuning:** Unfreezing the top $N$ layers of the base model and training with a very low learning rate ($1\text{e-}5$) so filters adapt directly to brain MRI textures.

---

### Technical Follow-Up & Limitations (36–42)

#### 36. Why didn't you unfreeze MobileNetV2 from the very beginning?
Because backpropagating massive gradients from an uninitialized head destroys ImageNet pre-trained weights in the first epoch (catastrophic forgetting).

#### 37. What happens if a user uploads a knee or chest X-ray?
Because the model has a closed-world assumption (Softmax across 4 classes), it will still output probabilities summing to 1.0, likely misclassifying it with low confidence. A production system must include an **Out-Of-Distribution (OOD)** input rejection classifier.

#### 38. How would you handle 3D MRI volumes instead of 2D slices?
Switch to 3D CNN architectures (e.g. 3D ResNet, UNet, or Vision Transformers like Swin UNETR) operating on $(H \times W \times D)$ voxel arrays with 3D convolutions.

#### 39. What is the clinical risk of a False Negative in our system?
A patient with an early-stage glioma would be told their brain is healthy (`notumor`), delaying craniotomy or chemotherapy until the tumor causes irreversible neurological damage.

#### 40. Why did meningioma recall improve by 15% when retraining on the full dataset?
Adding the 1,120 held-out validation images exposed the classification head to wider variances in skull convexity and dural-tail borders, teaching it not to confuse meningiomas with pituitary lesions.

#### 41. How would you deploy this model to edge devices (e.g., portable field MRI scanners)?
Quantize the model using **TensorFlow Lite (TFLite)** to 8-bit integers (INT8 post-training quantization), reducing model size to ~3 MB and latency to under 10 ms on embedded ARM processors.

#### 42. Is this system ready for autonomous clinical diagnosis?
**No.** Deep learning models are diagnostic second-opinion aids. It is not FDA-approved and all outputs must be corroborated by a board-certified neuroradiologist.
"""

print("Writing Chapters 13 and 14 module...")
