"""
Chapters 4 and 5 for the End-to-End Documentation Guide.
"""

CHAPTER_4 = """# Chapter 4: Python Modules, Packages, and Imports

## 4.1 Theoretical Foundation: Modules, Packages, Libraries, Classes, Functions, and Objects

To speak like a seasoned software engineer, you must clearly distinguish these six core programming concepts:

1. **Module:** A single Python file (`.py`) containing executable code, functions, classes, and variables. For example, `app.py` is a module.
2. **Package:** A directory containing multiple Python modules, organized hierarchically, and traditionally accompanied by an `__init__.py` file. For example, `tensorflow` or `sklearn` are packages containing dozens of specialized sub-modules.
3. **Library:** A broader umbrella term for a comprehensive collection of packages and modules designed to solve a broad problem domain (e.g., OpenCV for Computer Vision, TensorFlow for Deep Learning).
4. **Class:** A blueprint or template for creating objects, defining data attributes and behavior methods (e.g., `tf.keras.layers.Dense` or `PIL.Image.Image`).
5. **Function:** A reusable, named block of code that takes zero or more inputs (parameters), performs computation, and optionally returns an output (e.g., `compute_gradcam()`).
6. **Object:** A concrete in-memory instance created from a class blueprint (e.g., `model = tf.keras.models.load_model(...)` creates an in-memory Model object).

---

## 4.2 Comprehensive Import Inspection Table

Here is every import statement across `app.py`, `01_data_exploration.ipynb`, and `02_model_training.ipynb`:

| Import Statement | Source Type | Library/Package | Purpose in Project | What Happens If Removed? | Plausible Alternatives |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `import os` | Python Standard Library | `os` | Operating system interactions (directory navigation, path parsing). | File path lookups and filesystem queries fail with `NameError: name 'os' is not defined`. | `pathlib`, `sys` |
| `from pathlib import Path` | Python Standard Library | `pathlib` | Object-oriented filesystem path resolution with cross-platform slash handling (`/` vs `\`). | Path manipulations crash across Windows/Linux environments. | `os.path` |
| `import glob` | Python Standard Library | `glob` | Pattern-based wildcard file search (e.g., `glob.glob("dataset/Testing/glioma/*.jpg")`). | Cannot dynamically discover and list test images in directories. | `os.scandir()`, `os.listdir()`, `Path.glob()` |
| `import hashlib` | Python Standard Library | `hashlib` | Computes MD5 cryptographic checksums for image deduplication and leakage auditing. | Cannot verify dataset hygiene or detect duplicate images between train and test. | `xxhash`, `zlib.crc32` |
| `import random` | Python Standard Library | `random` | Sets pseudo-random seeds and samples random images for visualization grids. | Sample image grids cannot be generated reproducibly. | `numpy.random` |
| `from collections import Counter` | Python Standard Library | `collections` | High-performance frequency counting for image dimensions, aspect ratios, and color modes. | Counting image resolutions requires manual, slow dictionary iteration. | Standard `dict` loops |
| `import numpy as np` | External Third-Party | `numpy` | Vectorized numerical computing, array reshaping, tensor slicing, and `argmax` probability decoding. | The entire mathematical backbone of the project collapses immediately. | None (NumPy is the universal standard) |
| `import pandas as pd` | External Third-Party | `pandas` | Tabular data structures (`DataFrame`) for dataset counts, class balances, and audit records. | Summary statistics and audit tables must be formatted manually as raw dictionaries. | `polars`, Python `csv` |
| `from PIL import Image` | External Third-Party | `Pillow (PIL)` | High-level image loading, decoding raw byte streams from Streamlit file uploaders, and format verification. | Streamlit cannot ingest or decode uploaded user image files into memory. | `imageio`, `cv2.imdecode` |
| `import cv2` | External Third-Party | `opencv-python` | Image resizing (bilinear interpolation), BGR-to-RGB color conversion, and applying `COLORMAP_JET` to Grad-CAM heatmaps. | Cannot apply thermal heatmaps to Grad-CAM or resize NumPy arrays fast. | `matplotlib.cm`, `scikit-image` |
| `import matplotlib.pyplot as plt` | External Third-Party | `matplotlib` | Generating static research plots, loss/accuracy curves, and multi-image diagnostic comparison grids. | Cannot render learning curves or multi-panel diagnostic figures in notebooks. | `seaborn`, `plotly` |
| `import seaborn as sns` | External Third-Party | `seaborn` | Statistical visualization, specifically rendering annotated, color-scaled Confusion Matrix heatmaps. | Confusion matrices can only be viewed as raw numeric text grids. | `matplotlib.pyplot.matshow` |
| `from sklearn.metrics import classification_report, confusion_matrix` | External Third-Party | `scikit-learn` | Computes precision, recall, F1-scores, support counts, and the 4x4 confusion matrix array. | Clinical performance metrics cannot be calculated without writing manual math formulas. | Custom manual calculation |
| `import tensorflow as tf` | External Third-Party | `tensorflow` | Deep Learning computational framework: compiles computational graphs, executes GPU/CPU kernels, and runs neural networks. | The neural network cannot be trained, loaded, or executed. | `PyTorch`, `JAX` |
| `from tensorflow.keras import layers, models` | External Third-Party | `tensorflow` (Keras API) | Constructs neural network architectures via functional API: `Input`, `Dense`, `Dropout`, `BatchNormalization`, `Model`. | Cannot build the classification head or compose the deep learning pipeline. | Raw `tf.nn` operations |
| `from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau` | External Third-Party | `tensorflow.keras` | Training lifecycle hooks: automates saving best weights to disk, stops training on loss saturation, and decays learning rate. | Risk of severe model overfitting; best epoch weights are lost when training finishes. | Manual training loop with Python `if` statements |
| `import streamlit as st` | External Third-Party | `streamlit` | Reactive web application framework for building browser dashboards with pure Python. | The user interface cannot run. | `FastAPI + React`, `Flask + HTML/JS` |

---

## 4.3 Why Dependency Management & `requirements.txt` Matter

In professional software engineering, **environment reproducibility** is paramount. If a colleague attempts to run this project on another computer using TensorFlow 2.11 or NumPy 2.0, catastrophic failures occur:
- **NumPy 2.0 C-API Breaking Changes:** TensorFlow binaries compiled against NumPy 1.x crash immediately when loaded with NumPy 2.0 (`ImportError: numpy.core.multiarray failed to import`).
- **Keras 3 vs Keras 2 Discrepancies:** TensorFlow 2.16+ switched to Keras 3 as the default engine, changing how internal layers, training state, and functional graphs serialize.

By pinning versions in `requirements.txt` (e.g., `numpy>=1.24.0,<2.0.0` and `tensorflow>=2.15.0`), we lock the software ecosystem into a stable, tested matrix where all C-extensions and serialization APIs function seamlessly.
"""

CHAPTER_5 = """# Chapter 5: Python Concepts Used in the Project

## 5.1 Concept Inventory

This project utilizes 12 essential Python concepts. Below, each concept is dissected with:
1. Simple Definition
2. Five-Year-Old Child Analogy
3. Small Generic Code Example
4. Actual Usage in Our Source Code
5. Why It Is Needed
6. Failure Mode If Removed or Altered

---

### 1. Variables and Data Types
- **Definition:** Named storage locations in computer memory that hold specific types of data (integers, floating-point numbers, strings, booleans, lists, tuples, and multi-dimensional NumPy arrays).
- **5-Year-Old Analogy:** Think of labeled toy boxes: one box only holds soft teddy bears (strings), another holds numbered building blocks (integers), and a third holds water cups filled to exact levels (floating-point numbers).
- **Small Example:**
  ```python
  image_size = (224, 224)  # Tuple of integers
  class_name = "glioma"     # String
  confidence = 0.942        # Floating-point number
  is_tumor = True           # Boolean
  ```
- **Actual Project Usage:** In `app.py`:
  ```python
  CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
  confidence = probs[pred_idx] * 100
  is_healthy = (pred_class == "notumor")
  ```
- **Why It Is Needed:** Provides identifiers to track data dynamically throughout computation.
- **Failure Mode:** Hardcoding raw numbers and strings everywhere makes code unmaintainable and prone to typos.

---

### 2. Functions, Parameters, and Return Values
- **Definition:** Reusable blocks of logic that accept inputs (parameters), perform operations, and return output values using the `return` keyword.
- **5-Year-Old Analogy:** Like a bread toaster: you put in a slice of white bread (parameter), push the lever (function computation), and out pops warm toast (return value).
- **Small Example:**
  ```python
  def calculate_percentage(score: float, total: float) -> float:
      return (score / total) * 100.0
  ```
- **Actual Project Usage:** In `app.py`:
  ```python
  def create_superimposed_overlay(original_rgb, heatmap, alpha=0.45):
      h, w, _ = original_rgb.shape
      heatmap_resized = cv2.resize(heatmap, (w, h))
      heatmap_uint8 = np.uint8(255 * heatmap_resized)
      jet_colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
      jet_rgb = cv2.cvtColor(jet_colormap, cv2.COLOR_BGR2RGB)
      superimposed = np.uint8(original_rgb * (1.0 - alpha) + jet_rgb * alpha)
      return superimposed, jet_rgb
  ```
- **Why It Is Needed:** Eliminates duplicate code and encapsulates complex multi-step image transformations into clean, testable units.
- **Failure Mode:** Without functions, image blending code would need to be re-typed every time an image is shown, leading to bugs.

---

### 3. Classes and Object-Oriented Programming (OOP)
- **Definition:** A programming paradigm based on creating "classes" (blueprints) that bundle data attributes with specific functions ("methods").
- **5-Year-Old Analogy:** A cookie cutter is a class; the delicious gingerbread cookies you bake using that cutter are objects. Every cookie has arms, legs, and icing, but you can decorate each one differently.
- **Small Example:**
  ```python
  class PatientScan:
      def __init__(self, patient_id: str, scan_type: str):
          self.patient_id = patient_id
          self.scan_type = scan_type
          
      def get_summary(self):
          return f"Patient {self.patient_id} ({self.scan_type})"
  ```
- **Actual Project Usage:** In `notebooks/02_model_training.ipynb`:
  ```python
  from tensorflow.keras import layers, models
  
  # Instantiating Keras Layer objects from Layer classes
  gap_layer = layers.GlobalAveragePooling2D(name="global_avg_pooling")
  dropout_layer = layers.Dropout(0.3, name="dropout_1")
  ```
- **Why It Is Needed:** Keras models and layers are structured as OOP objects holding internal trainable state (weights and biases).
- **Failure Mode:** Without OOP, tracking millions of floating-point weights across dozens of neural network layers would require thousands of loose global arrays.

---

### 4. Decorators (`@decorator_name`)
- **Definition:** A special Python syntax that wraps a function, modifying or extending its behavior without permanently altering the function's internal code.
- **5-Year-Old Analogy:** Like putting a waterproof raincoat on a child before they go outside: the child is still the exact same child, but now they have the superpower of staying dry in the rain!
- **Small Example:**
  ```python
  def announce(func):
      def wrapper(*args, **kwargs):
          print("Starting task...")
          result = func(*args, **kwargs)
          print("Task finished!")
          return result
      return wrapper

  @announce
  def greet(name):
      print(f"Hello, {name}!")
  ```
- **Actual Project Usage:** In `app.py`:
  ```python
  @st.cache_resource(show_spinner=False)
  def load_neuroscan_model():
      model = tf.keras.models.load_model(model_path)
      base_model = [l for l in model.layers if "mobilenet" in l.name.lower()][0]
      return model, base_model
  ```
- **Why It Is Needed:** Tells Streamlit: *"Run this expensive 12 MB model load operation ONCE. Keep the resulting object in RAM. On all future page refreshes, return the cached object instantly."*
- **Failure Mode:** Removing `@st.cache_resource` forces TensorFlow to reload the model from disk every time the user clicks a button, introducing an unacceptable 5-second lag per click.

---

### 5. Context Managers (`with` statement)
- **Definition:** A structure that manages the allocation and release of resources, ensuring setup and cleanup code runs reliably (even if errors occur).
- **5-Year-Old Analogy:** Like an automatic door at a supermarket: it opens the moment you walk near it, and closes safely behind you the second you walk through, so the air conditioning never escapes.
- **Small Example:**
  ```python
  with open("sample.txt", "r") as file:
      data = file.read()
  # File is automatically closed here, even if reading failed!
  ```
- **Actual Project Usage:** In `app.py`:
  ```python
  with tf.GradientTape() as tape:
      conv_features = base_model(prep_img, training=False)
      tape.watch(conv_features)
      ...
  ```
  And for UI layouts:
  ```python
  with col_left:
      st.markdown("#### Visual Saliency (Grad-CAM)")
      st.image(superimposed_img)
  ```
- **Why It Is Needed:** For `tf.GradientTape`, it records forward operations onto a computational tape for automatic differentiation. For Streamlit columns, it routes UI elements into specific visual columns.
- **Failure Mode:** Manual gradient tracking or manually clearing tape memory leads to memory leaks and unrecorded backpropagation graphs.

---

### 6. List Comprehensions
- **Definition:** A concise, expressive syntax for creating a new list by transforming or filtering elements of an existing iterable.
- **5-Year-Old Analogy:** Like walking past a basket of mixed fruit and picking out only the red strawberries, placing them neatly onto your plate in a single swift motion.
- **Small Example:**
  ```python
  scores = [0.1, 0.4, 0.8, 0.95]
  percentages = [s * 100 for s in scores]
  # Result: [10.0, 40.0, 80.0, 95.0]
  ```
- **Actual Project Usage:** In `app.py`:
  ```python
  base_model = [l for l in model.layers if "mobilenet" in l.name.lower()][0]
  ```
- **Why It Is Needed:** Finds the specific base model layer in one readable line without writing a clunky multi-line loop with temporary tracking variables.
- **Failure Mode:** Writing five lines of nested loops makes code verbose and increases risk of `IndexError`.

---

### 7. Dictionary Unpacking and Key-Value Mapping
- **Definition:** Storing and accessing data using associative key-value pairs (`dict`), enabling $O(1)$ constant-time lookup.
- **5-Year-Old Analogy:** Like a coat-check room at a theatre: you give the attendant your coat number (key), and they instantly hand you your specific coat (value).
- **Small Example:**
  ```python
  class_colors = {"glioma": "#ef4444", "notumor": "#10b981"}
  print(class_colors["glioma"])  # Returns '#ef4444'
  ```
- **Actual Project Usage:** In `app.py`:
  ```python
  CLASS_DISPLAY = {
      "glioma": "Glioma Tumor",
      "meningioma": "Meningioma Tumor",
      "notumor": "Healthy (No Tumor)",
      "pituitary": "Pituitary Tumor"
  }
  ```
- **Why It Is Needed:** Decouples internal dataset folder names (`notumor`) from human-friendly clinical presentation titles (`Healthy (No Tumor)`).
- **Failure Mode:** Without dictionaries, code requires chains of `if/elif/else` statements across every display function.

---

### 8. Exception Handling (`try...except`) and Conditional Guards
- **Definition:** Defensive programming constructs that catch runtime errors and prevent program crashes when encountering unexpected states.
- **5-Year-Old Analogy:** Wearing knee pads and a helmet while riding a bicycle: if you hit a pebble and stumble, you don't break a bone—you brush yourself off and keep riding safely.
- **Small Example:**
  ```python
  try:
      number = int("invalid_number")
  except ValueError:
      number = 0
  ```
- **Actual Project Usage:** In `app.py`:
  ```python
  if not model_path.exists():
      st.error(f"Model file not found at: {model_path}")
      return None, None
  ```
- **Why It Is Needed:** Prevents the web server from throwing an ugly, scary Python traceback to a clinician if a file path is misconfigured.
- **Failure Mode:** Uncaught exceptions immediately crash the Streamlit web process and terminate the application.

---

### 9. Type Hints and Annotations
- **Definition:** Explicit type markers indicating what data types a function expects to receive and return (e.g., `def func(x: int) -> str:`).
- **5-Year-Old Analogy:** Like a square peg hole on a wooden puzzle board: it shows you that a square peg belongs here, even though a wooden hammer could technically force a round peg through.
- **Small Example:**
  ```python
  def resize_scan(image_array: np.ndarray, target_dim: tuple[int, int]) -> np.ndarray:
      return cv2.resize(image_array, target_dim)
  ```
- **Important Technical Truth:** In standard Python, **type hints do NOT enforce types at runtime**. Python is dynamically typed. Passing a string to an argument annotated as `int` will NOT throw a TypeError unless runtime validation libraries like Pydantic are used. Type hints exist for developer clarity, IDE autocompletion, and static linters (`mypy`).
- **Why It Is Needed:** Helps developers immediately understand data structures without digging through third-party documentation.

---

### 10. Virtual Environments (`.venv`)
- **Definition:** An isolated directory tree containing a dedicated Python executable and private copy of `site-packages`.
- **5-Year-Old Analogy:** Having your own private coloring pencil box with your name on it, so your little brother cannot swap your sharp red pencil for a dull broken crayon.
- **Actual Project Usage:** The `.venv` folder in `D:\Brain_Tumor`.
- **Why It Is Needed:** Isolates TensorFlow 2.15 and NumPy 1.26 from other projects on the machine that might require PyTorch or NumPy 2.0.
- **Failure Mode:** Global package upgrades on the host machine break neural network checkpoint deserialization.
"""

print("Writing Chapters 4 and 5 module...")
