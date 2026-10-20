"""
Chapters 1, 2, and 3 for the End-to-End Documentation Guide.
"""

CHAPTER_1 = """# Chapter 1: Project Introduction

## 1.1 Project Overview & Official Identity
- **Project Name:** NeuroScan AI — Brain Tumor Detection & Diagnostic Studio
- **Domain:** Medical Computer Vision, Clinical Decision Support Systems (CDSS), Explainable Artificial Intelligence (XAI)
- **Primary Goal:** To provide real-time, four-class intracranial tumor diagnosis from magnetic resonance imaging (MRI) scans with integrated gradient-weighted spatial explainability (Grad-CAM).

---

## 1.2 Problem Statement & Clinical Relevance
Brain tumors represent some of the most lethal and diagnostically challenging neoplasms in modern oncology. Among primary central nervous system (CNS) lesions, three major histological and anatomical varieties predominate:
1. **Gliomas:** Malignant, highly infiltrative intra-axial tumors originating from glial supportive tissue (astrocytomas, oligodendrogliomas, glioblastoma multiforme). Because they intermingle with healthy neural parenchyma, early border identification is life-critical.
2. **Meningiomas:** Extra-axial intracranial tumors arising from the arachnoid cells of the meninges surrounding the brain. Although predominantly benign, their compressive mass effect can cause irreversible neurological deficits.
3. **Pituitary Adenomas:** Sellar neoplasms growing from the pituitary gland at the base of the skull, immediately adjacent to the optic chiasm. Left untreated, they cause endocrine failure and bilateral visual field blindness.
4. **Normal / Healthy Anatomy:** Absence of space-occupying intracranial lesions.

### The Diagnostic Bottleneck
Radiologists examine hundreds of high-resolution MRI slices daily. High diagnostic volume creates cognitive fatigue, leading to missed micro-lesions, inter-observer diagnostic variability, and delayed surgical intervention.

### The "Black-Box" AI Dilemma in Healthcare
While Deep Learning models can classify medical images with high mathematical precision, traditional neural networks are inscrutable "black boxes." A neural network might output: *"Meningioma: 95% Confidence"*, but offer no explanation for *why*. If the network secretly based its decision on scanner artifacts, patient motion blur, or skull clamp metal, treating physicians could make fatal surgical errors. 

**NeuroScan AI solves this exact problem by marrying high-accuracy transfer learning with mathematically grounded visual explainability (Grad-CAM).**

---

## 1.3 Who Uses NeuroScan AI?
1. **Neuroradiologists:** As a diagnostic second-opinion tool during high-volume screening shifts.
2. **Neurosurgeons:** For rapid visual orientation and verification of tumor margin regions prior to biopsy or craniotomy.
3. **Emergency Medicine Physicians:** In trauma centers to rapidly triage acute neurological admissions and rule out mass effect or herniation risk.
4. **Medical Imaging Researchers & Students:** As an open, transparent platform to study how convolutional filters respond to intracranial pathologies.

---

## 1.4 Main Objectives and Implemented Features

### Objectives
- Perform multi-class classification into four distinct radiological categories (`glioma`, `meningioma`, `notumor`, `pituitary`).
- Minimize false negative diagnoses for tumor presence, ensuring no patient with an active neoplasm is mistakenly discharged as healthy.
- Provide zero-cost visual localization of the tumor bed without requiring expensive, time-consuming manual pixel masks.
- Deliver an interactive, zero-latency clinical dashboard accessible via any modern web browser.

### Implemented Features
1. **Full-Pipeline Deep Learning Engine:** Built on TensorFlow/Keras using transfer learning with a MobileNetV2 backbone.
2. **Empirically Validated 86.12% Test Accuracy:** Evaluated across 1,600 untouched, previously unseen test MRI scans.
3. **99.0% Healthy Brain Recall:** Near-zero false alarms for tumor presence on normal brain scans.
4. **Real-Time Grad-CAM Saliency Studio:** Computes gradients at the final convolutional feature layer (`out_relu`) to generate thermal overlays highlighting the tumor.
5. **Interactive Streamlit Web Dashboard (`app.py`):** Features dual input modes (file upload + pre-loaded test gallery), multi-class probability bars, and sub-second in-memory model caching.

---

## 1.5 Technology Stack

| Layer | Technology | Version | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11.x | Core programming language |
| **Deep Learning** | TensorFlow / Keras | >= 2.15.0 | Neural network construction, training loop, and Grad-CAM backpropagation |
| **Scientific Computing** | NumPy | >= 1.24.0, < 2.0.0 | Multi-dimensional tensor operations, array manipulation, and probability math |
| **Data Analysis** | Pandas | >= 2.0.0 | Tabular dataset tracking and audit record storage |
| **Computer Vision** | OpenCV (`opencv-python`) | >= 4.8.0 | Image resizing, color-space conversion (BGR/RGB), and Jet colormap generation |
| **Image IO** | Pillow (PIL) | >= 9.5.0 | High-fidelity image decoding and Streamlit file stream handling |
| **Evaluation Metrics** | Scikit-Learn | >= 1.3.0 | Multi-class classification report, precision, recall, and confusion matrix |
| **Visualizations** | Matplotlib & Seaborn | >= 3.7.0 / >= 0.12.0 | Training convergence curves and confusion matrix heatmaps |
| **Web Frontend** | Streamlit | >= 1.30.0 | Reactive, in-memory clinical dashboard with dark-mode glassmorphic styling |

---

## 1.6 Inputs and Outputs

```
+-------------------------------------------------------------+
|                        INPUT DATA                           |
|  - Axial, Coronal, or Sagittal Cranial MRI Slice            |
|  - Formats: JPEG, JPG, PNG                                  |
|  - Dimensions: Arbitrary (Rescaled automatically to 224x224)|
|  - Channels: Grayscale (L), RGB, or RGBA (Normalized to RGB)|
+-------------------------------------------------------------+
                              |
                              v
                  [ NEUROSCAN AI PIPELINE ]
                              |
                              v
+-------------------------------------------------------------+
|                       OUTPUT RESULTS                        |
|  1. Primary Diagnostic Class (e.g., "Pituitary Tumor")       |
|  2. Confidence Percentage (e.g., "99.4%")                   |
|  3. Multi-Class Probability Breakdown (All 4 classes)       |
|  4. Grad-CAM Attention Heatmap (Thermal activation map)     |
|  5. Superimposed Diagnostic Overlay (Scan + Glowing Marker) |
+-------------------------------------------------------------+
```

---

## 1.7 Current Project Limitations
1. **2D Slice Processing:** The model operates on single 2D slices rather than fully volumetric 3D DICOM volumes (e.g., multi-planar 3D T1-contrast, T2-FLAIR sequences).
2. **Glioma vs. Meningioma Boundary Confusion:** Because the MobileNetV2 base feature extractor remains frozen, the network occasionally confuses infiltrative glioma tissue with dural-tail meningiomas (67 test gliomas were predicted as meningiomas).
3. **Regulatory Status:** The system is an investigational and educational software system, not an FDA-approved medical diagnostic device.

---

## 1.8 Explaining the Project to Three Different Audiences

### 1. Like Explaining to a Five-Year-Old Child
> "Imagine your brain is a beautiful, peaceful playground. Sometimes, an unwelcome visitor—like a naughty, sneaky muddy monster—hides inside the playground. A doctor takes a picture of the playground, but the picture is black, gray, and white, making it really hard to spot the muddy monster! 
> 
> Our project is like a super-smart robot owl named **NeuroScan**. NeuroScan has looked at thousands of playground pictures before. When you show it a new picture, it immediately shouts, *'Aha! There is a monster hiding in the corner near the swing set!'* And even better, NeuroScan uses a magic glowing orange paintbrush to color directly over the monster so the doctor knows exactly where to look!"

### 2. Like Explaining to a Beginner with Basic Computer Knowledge
> "NeuroScan AI is a website and computer vision program that identifies brain tumors from MRI scans. When a hospital takes a brain MRI, it is saved as an image file. 
> 
> You simply drag and drop that image onto our webpage. Behind the scenes, a neural network—which is an artificial intelligence model trained on 5,600 verified brain scans—inspects the pixels. It calculates the odds of four possibilities: Glioma, Meningioma, Pituitary tumor, or Healthy Brain. 
> 
> Within half a second, the screen displays the most likely diagnosis with a percentage score. Next to the picture, the program draws a glowing thermal heatmap over the scan. This heatmap works like an X-ray highlighter: red and yellow spots show the exact pixels that convinced the AI that a tumor exists."

### 3. Like Explaining to a Technical Interviewer
> "NeuroScan AI is an end-to-end medical image classification and interpretability platform implemented in Python using TensorFlow/Keras and Streamlit. The architecture utilizes transfer learning with a MobileNetV2 backbone pretrained on ImageNet, coupled with a custom dense classification head incorporating Global Average Pooling, Batch Normalization, and Dropout for regularization. 
> 
> The pipeline processes 224x224x3 cranial MRI scans across four classes. To overcome the black-box dilemma in clinical AI, we implemented a custom Grad-CAM module using TensorFlow's `GradientTape`. It backpropagates class-specific score gradients to the final convolutional layer (`out_relu`), computes global-average-pooled channel importance weights, and applies a rectified linear combination to construct a 2D saliency map. 
> 
> Empirically, training on the complete 5,600-image dataset boosted test accuracy on 1,600 untouched scans from an 82.19% baseline to 86.12%, with meningioma sensitivity jumping from 63.5% to 78.5% and healthy brain recall achieving 99.0%. The production model is deployed via a reactive Streamlit application featuring in-memory model caching via `@st.cache_resource`."

---

## 1.9 A Realistic End-to-End Clinical Usage Scenario

**Setting:** Department of Emergency Radiology, St. Jude Regional Hospital.  
**Time:** 02:15 AM.  
**Patient:** A 52-year-old female arrives with acute severe cephalalgia (headache), left-sided hemiparesis, and disorientation.  
**Step 1:** The MRI technician runs a rapid axial cranial scan. The scanner outputs an axial T1 slice showing a mass near the right frontoparietal convexity.  
**Step 2:** The resident radiologist opens **NeuroScan AI** in the browser (`http://localhost:8501`).  
**Step 3:** The resident drags the MRI slice into the file uploader.  
**Step 4:** Within 400 milliseconds:
- The Primary Diagnosis Banner lights up: `⚠️ Meningioma Tumor — 91.4% Confidence`.
- The Multi-Class Probability distribution indicates:
  - Meningioma: 91.42%
  - Glioma: 6.81%
  - Pituitary: 1.63%
  - Healthy (No Tumor): 0.14%
- The **Grad-CAM Saliency Studio** renders an intense red-yellow hotspot along the extra-axial dural margin of the right frontoparietal lobe.
**Step 5:** The resident verifies that the network is focusing directly on the dural attachment and not skull bone or ventricular fluid. Confident in the automated screening corroboration, the resident immediately pages the on-call neurosurgeon for surgical decompression.
"""

CHAPTER_2 = """# Chapter 2: Project Folder Structure

## 2.1 Complete Directory Tree
Here is the exact filesystem structure of the `d:\\Brain_Tumor` workspace:

```text
d:\Brain_Tumor
├── .git/                                               # Git version control repository database
├── .gitignore                                          # Specifies files ignored by version control
├── .venv/                                              # Isolated Python 3.11 virtual environment
├── dataset/                                            # Raw MRI dataset directory
│   ├── Testing/                                        # Untouched test set (1,600 images)
│   │   ├── glioma/                                     # 400 test images for glioma
│   │   ├── meningioma/                                 # 400 test images for meningioma
│   │   ├── notumor/                                    # 400 test images for healthy brains
│   │   └── pituitary/                                  # 400 test images for pituitary adenoma
│   └── Training/                                       # Full training set (5,600 images)
│       ├── glioma/                                     # 1,400 training images
│       ├── meningioma/                                 # 1,400 training images
│       ├── notumor/                                    # 1,400 training images
│       └── pituitary/                                  # 1,400 training images
├── models/                                             # Serialized neural network checkpoints
│   ├── brain_tumor_mobilenetv2_best.keras             # Baseline checkpoint from 80/20 run (11.1 MB)
│   └── brain_tumor_mobilenetv2_full_scratch.keras     # Production checkpoint from full retrain (11.1 MB)
├── notebooks/                                          # Jupyter research & experimentation notebooks
│   ├── 01_data_exploration.ipynb                       # Dataset auditing, resolution check, MD5 leak audit
│   └── 02_model_training.ipynb                         # Model architecture, training loops, evaluation, Grad-CAM
├── app.py                                              # Streamlit web application & clinical dashboard
├── README.md                                           # Comprehensive GitHub project presentation document
└── requirements.txt                                    # Explicitly pinned third-party package dependencies
```

---

## 2.2 Deep Breakdown of Every File and Folder

### 1. `app.py`
- **What it contains:** The complete Python source code for the interactive Streamlit medical dashboard. It includes model loading, caching decorators, image tensor preprocessing, the Grad-CAM calculation engine, OpenCV color blending, and the reactive UI layout.
- **Why it exists:** Serves as the user-facing application and production deployment entry point.
- **Responsibility:** Accepts user image uploads, executes inference and explainability routines, and renders visual diagnostics.
- **Which other files use it:** End-users run it via `streamlit run app.py`. It imports weights from `models/` and samples from `dataset/Testing/`.
- **Essential or Optional?** **Essential.** This is the core application delivering user value.
- **Failure Mode if Missing:** Users cannot interact with the system via a graphical interface; inference could only occur through raw Python scripts.
- **Creation Method:** Manually written and iteratively refined.

### 2. `notebooks/01_data_exploration.ipynb`
- **What it contains:** Exploratory Data Analysis (EDA) code covering class distribution bar charts, image dimensions and aspect ratios, color channel distributions (RGB vs. L vs. RGBA), and MD5 cryptographic hash deduplication checks between training and testing sets.
- **Why it exists:** Machine learning rule #1 is "Understand your data before modeling." This notebook validates dataset integrity.
- **Responsibility:** Proves there is zero data leakage between train and test splits, verifies dataset balance (1,400 train / 400 test per class), and documents the need for 224x224 RGB resizing.
- **Which other files use it:** Informs the preprocessing decisions codified in `02_model_training.ipynb` and `app.py`.
- **Essential or Optional?** Essential for research verification, scientific auditability, and interview defense.
- **Failure Mode if Missing:** The team would lack documented proof of dataset integrity and leakage-free validation.
- **Creation Method:** Created interactively during initial exploratory research.

### 3. `notebooks/02_model_training.ipynb`
- **What it contains:** The complete deep learning experimental lifecycle: `tf.data` input pipeline construction, data augmentation Sequential layers, MobileNetV2 architecture assembly, callback definitions, baseline training (80/20 split), full-dataset retraining (100% split), classification reports, confusion matrix heatmaps, and test-scan Grad-CAM visualization.
- **Why it exists:** Serves as the laboratory where models are trained, evaluated, compared, and serialized.
- **Responsibility:** Produces the trained `.keras` weight files stored in `models/`.
- **Which other files use it:** Produces `models/brain_tumor_mobilenetv2_full_scratch.keras`, which `app.py` loads.
- **Essential or Optional?** **Essential.** Without this notebook, model weights cannot be reproduced or retrained.
- **Failure Mode if Missing:** Inability to retrain models, inspect training convergence curves, or alter hyperparameters.
- **Creation Method:** Created interactively during model development and experimentation.

### 4. `models/brain_tumor_mobilenetv2_full_scratch.keras`
- **What it contains:** The compiled production neural network artifact saved in native Keras v3 format. Contains model architecture topology, optimizer state, and all 2,422,084 floating-point weight parameters.
- **Why it exists:** Stores the mathematical intelligence learned from all 5,600 training images, achieving 86.12% test accuracy.
- **Responsibility:** Provides the pre-trained weights for instant inference in `app.py`.
- **Which other files use it:** Loaded by `app.py` via `tf.keras.models.load_model()`.
- **Essential or Optional?** **Essential.** Without this file, the application cannot perform inference without retraining for 15 minutes.
- **Failure Mode if Missing:** `app.py` raises an error: `Model file not found`.
- **Creation Method:** Automatically generated and saved by `ModelCheckpoint` in `02_model_training.ipynb`.

### 5. `models/brain_tumor_mobilenetv2_best.keras`
- **What it contains:** The baseline model artifact saved during the initial 80/20 validation split run (82.19% test accuracy).
- **Why it exists:** Kept as a comparative scientific benchmark and automated fallback in `app.py`.
- **Responsibility:** Serves as a historical baseline proving the empirical benefit of full retraining.
- **Which other files use it:** Referenced in `notebooks/02_model_training.ipynb` for evaluation; fallback in `app.py`.
- **Essential or Optional?** Optional in production, but critical for scientific comparison.
- **Failure Mode if Missing:** `app.py` continues running seamlessly using the primary full model.
- **Creation Method:** Automatically generated by `ModelCheckpoint` during the first training experiment.

### 6. `dataset/Testing/`
- **What it contains:** 1,600 verified MRI images strictly divided into four subfolders: `glioma/` (400), `meningioma/` (400), `notumor/` (400), and `pituitary/` (400).
- **Why it exists:** Provides an untouched evaluation set to measure true generalization. Also provides sample images for the web app's interactive gallery.
- **Responsibility:** Ground-truth testing split.
- **Which other files use it:** Evaluated in `02_model_training.ipynb` and scanned by `app.py` to populate the "Select Clinical Test Sample" dropdown.
- **Essential or Optional?** **Essential.** Tracked in Git so that cloned repositories have immediate test images.
- **Failure Mode if Missing:** Final test metrics cannot be computed, and the web app's pre-loaded sample picker will fail.
- **Creation Method:** Curated from the clinical MRI dataset.

### 7. `dataset/Training/`
- **What it contains:** 5,600 MRI images divided equally (1,400 per class) across four class folders.
- **Why it exists:** Provides the training corpus used during neural network parameter optimization.
- **Responsibility:** Feeds the convolutional layers during backpropagation.
- **Which other files use it:** Consumed exclusively by `notebooks/01_data_exploration.ipynb` and `notebooks/02_model_training.ipynb`.
- **Essential or Optional?** Essential for model training; ignored by Git (`.gitignore`) to keep repository size small.
- **Failure Mode if Missing:** The model cannot be trained from scratch. However, inference in `app.py` still functions because the model weights are pre-saved.
- **Creation Method:** Curated from the source clinical dataset.

### 8. `requirements.txt`
- **What it contains:** An exact list of all external Python libraries and their version constraints required to execute the notebooks and web application.
- **Why it exists:** Ensures reproducible execution across different machines, operating systems, and team members.
- **Responsibility:** Tells `pip` which packages to install.
- **Which other files use it:** Invoked by developers during environment setup (`pip install -r requirements.txt`).
- **Essential or Optional?** **Essential.**
- **Failure Mode if Missing:** New developers or production servers must guess package versions, causing dependency conflicts.
- **Creation Method:** Manually curated with version boundaries.

### 9. `.gitignore`
- **What it contains:** Rules telling Git which files and directories to exclude from version control tracking (e.g., `.venv/`, `dataset/Training/`, `__pycache__/`, `.ipynb_checkpoints/`).
- **Why it exists:** Prevents accidental uploading of virtual environments, cache files, and huge datasets that would bloat the GitHub repository past limits.
- **Responsibility:** Maintains Git repository hygiene and keeps clone size under 50 MB.
- **Which other files use it:** Read automatically by the Git CLI.
- **Essential or Optional?** **Essential.**
- **Failure Mode if Missing:** Git attempts to commit thousands of virtual environment binaries and 125 MB of training images, risking failed pushes.
- **Creation Method:** Manually written.

### 10. `README.md`
- **What it contains:** Comprehensive project presentation document written in GitHub Flavored Markdown, featuring badges, executive summary, architecture diagrams, benchmark comparison tables, mathematical Grad-CAM explanations, and quickstart commands.
- **Why it exists:** Serves as the public face of the project on GitHub.
- **Responsibility:** Informs reviewers, recruiters, and collaborators about how to run and understand the project.
- **Which other files use it:** Rendered automatically by GitHub on the repository homepage.
- **Essential or Optional?** **Essential** for portfolio presentation and professional open-source standards.
- **Failure Mode if Missing:** Visitors see a bare directory of code without context or instructions.
- **Creation Method:** Manually written and structured.

### 11. `.venv/`
- **What it contains:** The self-contained Python 3.11 virtual environment holding the isolated Python executable, standard library copies, and installed site-packages (TensorFlow, Streamlit, OpenCV, etc.).
- **Why it exists:** Prevents package version conflicts with the user's global system Python.
- **Responsibility:** Executes Python code using pinned project dependencies.
- **Which other files use it:** Used whenever running `python` or `streamlit` commands.
- **Essential or Optional?** Essential locally; never committed to Git (recreated via `requirements.txt`).
- **Failure Mode if Missing:** Code executes in system Python where packages may be missing or version-incompatible.
- **Creation Method:** Created via `python -m venv .venv`.
"""

CHAPTER_3 = """# Chapter 3: Project Architecture & Component Relationships

## 3.1 High-Level Real-World Analogy
Think of NeuroScan AI like a **Modern Specialized Hospital Eye-and-Brain Clinic**:

1. **The Reception Desk (Streamlit Frontend):**  
   A patient walks in carrying an MRI envelope. The receptionist receives the scan, verifies that it is a valid picture, and places it on a standardized digital tray.
2. **The Medical Records Archive (Model Caching):**  
   Instead of hiring and training a brand-new doctor every time a patient walks in, the clinic has its Chief Specialist permanently on-site in the consultation room. The specialist's brain is already loaded with medical knowledge.
3. **The Physician's Assistant (Preprocessing Engine):**  
   Before the doctor looks at the scan, the assistant resizes the film to exactly $224 \times 224$ centimeters and adjusts the lighting scale from $[0, 255]$ to $[-1, 1]$ so the specialist's eyes can interpret it instantly.
4. **The Chief Specialist (MobileNetV2 Neural Network):**  
   The specialist examines the scan. The visual cortex (the convolutional base) extracts shapes, edges, and tissue densities. The diagnostic brain center (the classification head) weighs the evidence and issues a probability assessment: *"95% probability of Meningioma"*.
5. **The Transparent Highlighter Projection (Grad-CAM Engine):**  
   To prove they didn't just guess, the doctor clicks a switch that shines an infrared thermal lamp onto the scan film. The lamp glows bright red directly over the abnormal tumor mass, showing the patient and family the exact physical reason for the diagnosis.
6. **The Diagnostic Display Monitor (UI Dashboard):**  
   The patient receives a clear, color-coded report showing the highlighted scan, the diagnosis, and the probability breakdown.

---

## 3.2 Technical Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                              USER INTERACTION LAYER                                |
|                                                                                   |
|   +--------------------------+                     +--------------------------+   |
|   | 📁 Upload Custom MRI      |                     | 🔬 Select Test Sample    |   |
|   |    (JPG, JPEG, PNG)      |                     |    (From dataset/Testing)|   |
|   +-------------+------------+                     +-------------+------------+   |
+-----------------|------------------------------------------------|----------------+
                  |                                                |
                  +-----------------------+------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        FRONTEND PRESENTATION LAYER (app.py)                        |
|                                                                                   |
|   • Streamlit Reactive UI Engine                                                  |
|   • Dark-Mode Glassmorphism CSS Styling                                           |
|   • Layout Orchestration: Header -> Status Banner -> Dual Diagnostic Columns       |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     IN-MEMORY MODEL CACHING & INITIALIZATION                      |
|                                                                                   |
|   @st.cache_resource(show_spinner=False)                                          |
|   def load_neuroscan_model():                                                     |
|       model = tf.keras.models.load_model(FULL_SCRATCH_PATH)                       |
|       base_model = model.get_layer("mobilenetv2_1.00_224")                        |
|       return model, base_model                                                    |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         IMAGE PREPROCESSING PIPELINE                              |
|                                                                                   |
|   Input: Raw Image (PIL.Image or File Stream)                                     |
|   Step 1: Convert to 3-Channel RGB (img.convert('RGB'))                           |
|   Step 2: Bilinear Interpolation Resize to (224, 224) via OpenCV                  |
|   Step 3: Expand Batch Dimension -> Tensor Shape: (1, 224, 224, 3) (float32)      |
|   Step 4: Scale Pixels to [-1, 1] via tf.keras.applications.mobilenet_v2          |
+-----------------------------------------|-----------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
+------------------------------------+       +------------------------------------+
|     PRIMARY INFERENCE PATHWAY      |       |      GRAD-CAM SALIENCY ENGINE      |
|                                    |       |                                    |
| • Feed tensor to Functional Model  |       | • Track tensor in tf.GradientTape  |
| • Forward pass through Base Model  |       | • Extract feature maps from        |
| • GlobalAveragePooling2D           |       |   layer: base_model.output         |
| • Dense(128) Head + Regularization |       |   Shape: (1, 7, 7, 1280)           |
| • Dense(4, Softmax) Output         |       | • Forward pass through Head        |
|                                    |       | • Compute gradient:                |
| Result: Vector of 4 Probabilities  |       |   d(class_score) / d(conv_features)|
|   probs: [P_glio, P_men, P_no, P_pit]|     | • Global Average Pool Gradients    |
|   pred_idx = argmax(probs)         |       | • Linear Combination of Channels   |
|   confidence = probs[pred_idx] * 100|      | • Apply ReLU & Min-Max Normalize   |
|                                    |       | • Resize 7x7 -> 224x224 Heatmap    |
|                                    |       | • cv2.applyColorMap(COLORMAP_JET)  |
|                                    |       | • AddWeighted Overlay (alpha=0.45) |
+------------------------------------+       +------------------------------------+
                   |                                             |
                   +----------------------+----------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        COMPOSITED DIAGNOSTIC RENDERING                            |
|                                                                                   |
|   Left Column:                                 Right Column:                      |
|   [ 🔬 Visual Saliency Studio ]                [ 📊 Probability Distribution ]     |
|   • Tab 1: Superimposed Overlay (Scan + CAM)   • Sorted Color-Coded Progress Bars |
|   • Tab 2: Pure Jet Activation Heatmap         • Ground-Truth Match Indicator     |
|   • Tab 3: Side-by-Side Verification           • Diagnostic Confidence % Badge    |
+-----------------------------------------------------------------------------------+
```

---

## 3.3 Component Responsibilities & Communication Flow

### 1. Presentation Layer (`app.py`)
- **Receives:** User clicks, uploaded image file streams, or selected dropdown values.
- **Does:** Orchestrates the visual hierarchy. Checks if an image is loaded, converts byte streams into NumPy arrays, triggers the preprocessing and inference routines, and creates two responsive columns.
- **Returns:** Rendered HTML/CSS components and visual image elements to the browser DOM.

### 2. Model Caching Subsystem (`load_neuroscan_model`)
- **Receives:** Filepath constants pointing to `models/brain_tumor_mobilenetv2_full_scratch.keras`.
- **Does:** Invokes TensorFlow's model deserializer once on initial load. Decorates the loader with `@st.cache_resource`, instructing Streamlit to store the resulting Python object in global application memory.
- **Returns:** A tuple of `(model, base_model)` that persists across user sessions without reloading.

### 3. Preprocessing Engine
- **Receives:** An arbitrary input image (which could be $512 \times 512$ grayscale, RGBA, or $180 \times 180$).
- **Does:**
  1. Enforces three color channels (RGB) using `selected_image.convert("RGB")`.
  2. Resizes spatial dimensions to $(224, 224)$ via `cv2.resize()`.
  3. Casts data type from 8-bit unsigned integer (`uint8`, $[0, 255]$) to 32-bit floating point (`float32`).
  4. Expands tensor dimensions to add the batch axis: shape becomes `(1, 224, 224, 3)`.
  5. Scales pixel intensities to the interval $[-1.0, 1.0]$ using MobileNetV2's dedicated normalization formula: $\text{pixel}_{\text{norm}} = \frac{\text{pixel}}{127.5} - 1.0$.
- **Returns:** A standardized 4D tensor ready for mathematical operations.

### 4. Convolutional Feature Extractor (MobileNetV2 Base)
- **Receives:** The preprocessed tensor of shape `(1, 224, 224, 3)`.
- **Does:** Passes activations through 53 inverted residual convolutional blocks with depthwise separable filters. Low-level layers detect micro-edges and contrast gradients; middle layers detect contours and tissue boundaries; the final convolutional layer (`out_relu`) detects high-level semantic patterns characteristic of intracranial tumors.
- **Returns:** A feature activation map of shape `(1, 7, 7, 1280)`.

### 5. Classification Head
- **Receives:** The `(1, 7, 7, 1280)` feature map.
- **Does:**
  1. Collapses spatial dimensions via Global Average Pooling to a 1D vector of length 1,280.
  2. Normalizes channel statistics with Batch Normalization.
  3. Applies 30% Dropout to prevent co-adaptation.
  4. Computes 128 non-linear feature combinations using a Dense ReLU layer.
  5. Applies 20% Dropout.
  6. Computes final class unnormalized log-probabilities and passes them through a 4-way Softmax function.
- **Returns:** A probability distribution vector `probs` of length 4 summing to $1.0$.

### 6. Grad-CAM Explainability Engine (`compute_gradcam`)
- **Receives:** The input image tensor, the outer model, and the base model reference.
- **Does:**
  1. Opens a `tf.GradientTape` context.
  2. Watches the convolutional feature maps at `out_relu`.
  3. Forward-propagates activations through the classification head to isolate the score of the predicted class ($y^c$).
  4. Backpropagates to extract the gradients $\frac{\partial y^c}{\partial A^k}$.
  5. Spatially pools the gradients to compute channel importance weights $\alpha_k^c$.
  6. Takes the weighted sum of feature maps and clips negative values via ReLU.
  7. Normalizes values between $0.0$ and $1.0$.
- **Returns:** A 2D saliency heatmap array of shape `(7, 7)`.

### 7. Diagnostic Compositor (`create_superimposed_overlay`)
- **Receives:** The raw $224 \times 224$ RGB image array, the $7 \times 7$ float heatmap, and an overlay alpha parameter ($0.45$).
- **Does:**
  1. Upsamples the $7 \times 7$ heatmap to $224 \times 224$ using bilinear interpolation.
  2. Converts float values $[0.0, 1.0]$ to 8-bit unsigned integers $[0, 255]$.
  3. Maps single-channel intensities to an RGB thermal palette using OpenCV's `COLORMAP_JET` (high values = red/yellow, low values = deep blue).
  4. Blends the original scan with the thermal map: $\text{Pixel}_{\text{out}} = (1 - \alpha) \cdot \text{Scan} + \alpha \cdot \text{JetMap}$.
- **Returns:** A composite 3-channel image array highlighting the lesion site.
"""

print("Writing Chapters 1 to 3 module...")
