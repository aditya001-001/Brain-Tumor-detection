"""
Chapters 15 and 16 for the End-to-End Documentation Guide.
"""

CHAPTER_15 = """# Chapter 15: Explain the Project to a Five-Year-Old

## 15.1 The Story of Dr. Owl's Magic Brain Scanner

Once upon a time, deep in the Whispering Woods, there was a wonderful animal hospital run by a very wise doctor named **Dr. Owl**. 

Animals came from all over the forest when they had a headache. Dr. Owl had a special machine that took a photograph of the inside of an animal's head. The photograph showed the animal's brain, like a map of a magical kingdom!

### The Trouble in the Kingdom
Most of the time, the brain kingdom was happy and healthy, with clear rivers and peaceful paths. 

```text
       [ A Healthy Brain Kingdom ]          [ An Unwanted Mud Monster Hiding! ]
             .----------------.                       .----------------.
            /   (^_^)  (^_^)   \                     /   (^_^)   ###    \
           |   Peaceful brain   |                   |  Brain    #####   |  <-- "Tumor!"
           |    kingdom with    |                   |  kingdom  #####   |
            \   clear paths    /                     \         (^_^)   /
             '----------------'                       '----------------'
```

But sometimes, an unwanted guest called a **Mud Monster (a Tumor)** would sneak into the kingdom:
- Some mud monsters liked to hide inside the squishy brain soil—these were called **Gliomas**.
- Other mud monsters clung to the kingdom's outer protective brick walls—these were called **Meningiomas**.
- And tiny little monsters liked to hide in the deep royal basement under the castle—these were called **Pituitary Monsters**.

Because the photograph was only black, white, and gray, even the cleverest woodland doctors had trouble finding the sneaky mud monsters!

---

### Dr. Owl's Super-Team of Helpers

To help the sick animals, Dr. Owl built a super-team of magical assistants inside his treehouse:

#### 1. Pippin the Measuring Assistant
Whenever a rabbit brought in an MRI photograph, it came in all sorts of crazy sizes—some as big as a dinner plate, others tiny like a postage stamp! Pippin used a golden ruler to gently trim every photo so it was exactly $224 \times 224$ magic centimeters, and painted the colors so Dr. Owl's lenses could see it perfectly.

#### 2. Sherlock the Owl's Multi-Lens Glasses
Next, Dr. Owl slipped on his famous **Detective Glasses (MobileNetV2)**. These glasses had 53 special magnifying lenses stacked in a row:
- The first lens looked for simple straight sticks and round edges.
- The middle lenses looked for tissue textures and curves.
- The deepest lens looked for suspicious cloudy shapes where mud monsters hide!

#### 3. Barnaby the Scorekeeper
Standing next to Dr. Owl was Barnaby, holding four big chalkboard buckets labeled:
1. `Glioma`
2. `Meningioma`
3. `Pituitary`
4. `Healthy Brain`

Dr. Owl whispered what he saw through his lenses, and Barnaby dropped 100 golden marbles across the four buckets. If Barnaby dropped 95 marbles into the `Meningioma` bucket and only 5 in the others, he rang a bell: *"It's a Meningioma with 95% certainty!"*

#### 4. Sparky the Magic Paintbrush
*"Wait!"* cried the baby squirrel. *"How do we know you're not just guessing?"*

This was Dr. Owl's favorite moment. He pulled out **Sparky, the Magic Glowing Paintbrush (Grad-CAM)**. 
- Sparky flew backward through the 53 lenses of Dr. Owl's glasses to see which exact lens shouted the loudest when looking at the monster.
- Sparky dipped his bristles in bright fiery red and neon yellow paint, flying right onto the black-and-white photograph!
- **Swoosh!** A glowing red circle lit up right over the mud monster's head!

The baby squirrel gasped: *"Look! The magic paint is glowing right on the wall where the muddy bump is!"* 

Thanks to Dr. Owl, Pippin, Sherlock's glasses, Barnaby, and Sparky's magic paintbrush, the surgeon doctors knew exactly where to gently remove the mud monster, and the little woodland animals went home happy, healthy, and cured!

---

## 15.2 Mapping the Story to Our Actual Python Code

| Story Character | Role in the Fairy Tale | Real Component in Our Codebase | File Location |
| :--- | :--- | :--- | :--- |
| **The Hospital Treehouse** | Where all the animals meet and see the magic screen. | **Streamlit Web Application UI** | `app.py` |
| **The Patient's Photo** | The black-and-white map of the brain kingdom. | **Uploaded MRI Scan Image** | `app.py` (`st.file_uploader`) |
| **Pippin the Measuring Assistant** | Trims every photo to $224 \times 224$ and standardizes colors. | **Image Preprocessing Pipeline** | `app.py` (`cv2.resize`, `preprocess_input`) |
| **Sherlock's 53 Glasses** | Multi-lens glasses that recognize edges, contours, and lesions. | **MobileNetV2 Convolutional Base** | `app.py` (`mobilenetv2_1.00_224`) |
| **The 53rd Deepest Lens** | The final lens where semantic monster patterns are clearest. | **`out_relu` Conv Layer** (Shape: $7 \times 7 \times 1280$) | `base_model.output` |
| **Barnaby the Scorekeeper** | Drops 100 golden marbles across 4 buckets. | **Dense(4) Classification Head & Softmax** | `output_predictions` layer |
| **Sparky the Magic Paintbrush** | Traces backward through lenses to paint a glowing red circle. | **Grad-CAM Engine (`tf.GradientTape`)** | `compute_gradcam()` in `app.py` |
| **The Glowing Red Highlight** | The warm overlay showing the exact tumor site. | **Superimposed Jet Colormap Overlay** | `create_superimposed_overlay()` |
"""

CHAPTER_16 = """# Chapter 16: Glossary and Learning Roadmap

## 16.1 Comprehensive Technical Glossary

### 1. Convolutional Neural Network (CNN)
A class of deep neural networks specialized for visual data that uses mathematical convolution filters to capture spatial hierarchies and spatial relationships between pixels.
- *Project Example:* The MobileNetV2 backbone that inspects the $224 \times 224$ MRI scan.

### 2. Depthwise Separable Convolution
A factorized convolution that splits standard convolution into a depthwise spatial filter (one per channel) followed by a $1 \times 1$ pointwise channel-combination convolution.
- *Project Example:* Used across MobileNetV2's inverted residual blocks to reduce computation by ~8x.

### 3. Global Average Pooling 2D (GAP)
A dimensionality reduction technique that averages the 2D spatial height and width of each feature map into a single scalar value.
- *Project Example:* Converts the $(7 \times 7 \times 1280)$ output of `out_relu` into a 1D vector of length $1,280$.

### 4. Batch Normalization
A neural network layer that standardizes intermediate activations by subtracting the mini-batch mean and dividing by mini-batch standard deviation, accelerating convergence.
- *Project Example:* `layers.BatchNormalization(name="batch_norm_1")` in our classification head.

### 5. Dropout
A stochastic regularization technique where a random percentage of neuron outputs are zeroed during training to prevent feature co-adaptation.
- *Project Example:* `layers.Dropout(0.3)` and `layers.Dropout(0.2)` in our classification head.

### 6. Transfer Learning
A machine learning technique where a model pre-trained on a source task (e.g. ImageNet) is reused as the starting point for a model on a target task (e.g. Brain MRI).
- *Project Example:* Initializing MobileNetV2 with pre-trained ImageNet weights.

### 7. Catastrophic Forgetting
The abrupt forgetting of previously learned generic representations when a pre-trained network is updated with large gradients on a new, small dataset.
- *Project Example:* Prevented by setting `base_model.trainable = False`.

### 8. Data Leakage
When information from outside the training dataset (such as test samples) influences model training, leading to unrealistically optimistic validation metrics.
- *Project Example:* Audited and proven 100% absent in `01_data_exploration.ipynb` via MD5 checksum checks.

### 9. Categorical Cross-Entropy Loss
The standard loss function for multi-class classification problems with mutually exclusive target labels.
- *Project Example:* `tf.keras.losses.CategoricalCrossentropy()` in `02_model_training.ipynb`.

### 10. Softmax Function
An activation function that normalizes a vector of arbitrary raw numbers (logits) into a probability distribution where all values fall in $[0.0, 1.0]$ and sum to $1.0$.
- *Project Example:* `layers.Dense(4, activation="softmax")` producing our final 4 class probabilities.

### 11. Adam Optimizer
An adaptive learning rate optimization algorithm combining the advantages of momentum (first moment) and RMSprop (second moment).
- *Project Example:* Compiled with learning rate $0.001$ (`1e-3`).

### 12. Early Stopping
A training regularization technique that halts training when performance on a monitored metric (e.g., validation loss) ceases to improve over a set patience window.
- *Project Example:* `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`.

### 13. Model Checkpoint
A callback that saves the neural network architecture and weight tensors to disk whenever a new minimum loss is achieved.
- *Project Example:* Saved `models/brain_tumor_mobilenetv2_full_scratch.keras`.

### 14. Grad-CAM (Gradient-weighted Class Activation Mapping)
An explainable AI technique that computes gradients of the target class score with respect to convolutional feature maps to produce a coarse 2D localization heatmap.
- *Project Example:* `compute_gradcam()` in `app.py`.

### 15. `tf.GradientTape`
A Python context manager in TensorFlow that records tensor operations for automatic differentiation and gradient calculation.
- *Project Example:* Used in `compute_gradcam()` to extract $\frac{\partial y^c}{\partial A^k}$.

### 16. ColorMap JET
A standard rainbow thermal palette in OpenCV where low numbers are deep blue, medium numbers are cyan/green/yellow, and high numbers are bright red.
- *Project Example:* `cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)` for thermal overlays.

### 17. In-Memory Caching (`@st.cache_resource`)
A Streamlit decorator that caches global resources (such as database connections or 12 MB deep learning models) across user interactions and browser sessions.
- *Project Example:* Decorated `load_neuroscan_model()` in `app.py` to achieve sub-second response times.

### 18. Precision
The proportion of positive identifications that were actually correct: $\frac{TP}{TP + FP}$.
- *Project Example:* Glioma precision was $92.81\%$, meaning when the model diagnosed a glioma, it was right over 92% of the time.

### 19. Recall (Sensitivity)
The proportion of actual positives that were correctly identified: $\frac{TP}{TP + FN}$.
- *Project Example:* Healthy brain recall was $99.0\%$, meaning 396 out of 400 healthy scans were correctly identified.

### 20. F1-Score
The harmonic mean of Precision and Recall: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- *Project Example:* Weighted average F1-score across all 1,600 test scans was $0.8566$.

---

## 16.2 Step-by-Step Learning & Mastering Roadmap

If you want to understand every part of this project deeply and explain it with mastery in an interview, follow this 5-stage roadmap:

```
[ Stage 1: Data Understanding & Hygiene ] (Day 1)
   • Read: notebooks/01_data_exploration.ipynb
   • Master: Image loading, class distributions, resolution checks, and MD5 hash auditing.
   • Milestone: Be able to explain why train/test leakage destroys clinical AI models.
                     │
                     ▼
[ Stage 2: Deep Learning & Transfer Learning ] (Day 2-3)
   • Read: notebooks/02_model_training.ipynb (Cells 0 to 8)
   • Master: tf.data pipeline (cache, shuffle, prefetch), MobileNetV2 architecture,
             depthwise separable convolutions, GAP, BatchNorm, and Dropout.
   • Milestone: Explain why base_model.trainable was frozen initially.
                     │
                     ▼
[ Stage 3: Empirical Validation & Retraining ] (Day 4)
   • Read: notebooks/02_model_training.ipynb (Cells 9 to 15)
   • Master: Classification reports, Confusion Matrices, and the empirical jump
             from 82.19% to 86.12% when training on 100% of the dataset.
   • Milestone: Explain why meningioma recall surged by 15%.
                     │
                     ▼
[ Stage 4: Explainable AI with Grad-CAM ] (Day 5)
   • Read: app.py (compute_gradcam and create_superimposed_overlay)
   • Master: tf.GradientTape backpropagation, channel importance pooling, ReLU,
             and OpenCV Jet colormap alpha blending.
   • Milestone: Derive the Grad-CAM alpha_k equation on a whiteboard.
                     │
                     ▼
[ Stage 5: Production Deployment & Interview Mastery ] (Day 6-7)
   • Read: app.py (Streamlit UI layout, caching decorators) & README.md
   • Master: All 42 interview questions in Chapter 14.
   • Milestone: Deliver the 2-minute project walkthrough aloud without notes!
```
"""

print("Writing Chapters 15 and 16 module...")
