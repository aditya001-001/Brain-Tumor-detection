import os
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import tensorflow as tf

# ==============================================================================
# 1. PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Diagnostic System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. DESIGN SYSTEM & MODERN DARK-MODE CSS
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography & Root Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Header Banner */
    .header-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(120deg, #f8fafc 0%, #cbd5e1 60%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }

    /* Diagnostic Result Badges */
    .result-badge {
        border-radius: 12px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .badge-tumor {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border-color: rgba(239, 68, 68, 0.4);
    }

    .badge-healthy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border-color: rgba(16, 185, 129, 0.4);
    }

    .result-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
    }

    .result-confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Probability Bar Container */
    .prob-row {
        margin-bottom: 12px;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 4px;
        color: #cbd5e1;
    }
    .prob-track {
        height: 8px;
        background: rgba(51, 65, 85, 0.6);
        border-radius: 9999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Metric pill */
    .metric-pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. MODEL & CACHING PIPELINE
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "Healthy (No Tumor)",
    "pituitary": "Pituitary Tumor"
}

CLASS_DESCRIPTIONS = {
    "glioma": {
        "badge_color": "#ef4444",
        "description": "Intra-axial brain tumor arising from glial cells. These typically exhibit infiltrative margins invading surrounding brain parenchyma.",
        "urgency": "High Clinical Priority — Prompt Neurosurgical Consultation Required"
    },
    "meningioma": {
        "badge_color": "#f59e0b",
        "description": "Extra-axial intracranial tumor arising from the arachnoid cells of the meninges. Characterized by well-circumscribed borders and dural attachment.",
        "urgency": "Moderate Clinical Priority — Contrast MRI & Surgical Evaluation Indicated"
    },
    "pituitary": {
        "badge_color": "#8b5cf6",
        "description": "Benign adenoma originating from the pituitary gland within the sella turcica at the skull base. May cause visual field deficits due to optic chiasm compression.",
        "urgency": "Endocrine & Neurosurgical Review Indicated"
    },
    "notumor": {
        "badge_color": "#10b981",
        "description": "No space-occupying intracranial mass, pathological enhancement, or mass effect detected across cerebral hemispheres.",
        "urgency": "Low Priority — Normal Anatomical Variant"
    }
}

@st.cache_resource(show_spinner=False)
def load_neuroscan_model():
    """Loads and caches the best trained model in memory."""
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

# ==============================================================================
# 4. GRAD-CAM EXPLAINABILITY ENGINE
# ==============================================================================
def compute_gradcam(img_tensor, model, base_model, pred_index=None):
    """
    Computes Grad-CAM heatmap using the last convolutional activations of MobileNetV2.
    """
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

def create_superimposed_overlay(original_rgb, heatmap, alpha=0.45):
    """Blends the Grad-CAM jet heatmap with the original image."""
    h, w, _ = original_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_colormap, cv2.COLOR_BGR2RGB)
    superimposed = np.uint8(original_rgb * (1.0 - alpha) + jet_rgb * alpha)
    return superimposed, jet_rgb

# ==============================================================================
# 5. SIDEBAR: CLINICAL CONTROLS & TEST SAMPLES
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Diagnostic Controls")
    
    input_source = st.radio(
        "Scan Input Source:",
        ["📁 Upload MRI Scan", "🔬 Select Clinical Test Sample"],
        index=1
    )
    
    selected_image = None
    actual_label = None

    if input_source == "🔬 Select Clinical Test Sample":
        st.markdown("---")
        st.markdown("**Pre-loaded Dataset Samples:**")
        test_dir = BASE_DIR / "dataset" / "Testing"
        
        sample_options = {}
        for cls in CLASS_NAMES:
            cls_folder = test_dir / cls
            if cls_folder.exists():
                images = sorted(glob.glob(str(cls_folder / "*.jpg")))[:3]
                for idx, img_p in enumerate(images):
                    sample_options[f"{CLASS_DISPLAY[cls]} (Sample #{idx+1})"] = (img_p, cls)

        if sample_options:
            chosen_key = st.selectbox("Choose Sample:", list(sample_options.keys()))
            sample_path, actual_label = sample_options[chosen_key]
            selected_image = Image.open(sample_path)
            st.caption(f"Ground Truth: `{CLASS_DISPLAY[actual_label]}`")
        else:
            st.warning("Testing folder not found.")
            
    else:
        st.markdown("---")
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Scan",
            type=["jpg", "jpeg", "png"],
            help="Upload an axial, coronal, or sagittal T1/T2 MRI brain scan."
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)

# ==============================================================================
# 6. MAIN CONTENT AREA
# ==============================================================================
# Header Card
st.markdown("""
<div class="header-card">
    <div class="header-badge">Clinical Deep Learning System</div>
    <h1 class="main-title">NeuroScan AI — Brain Tumor Diagnostic Studio</h1>
    <p class="subtitle">Upload an MRI scan or choose a test sample to get instant classification and Grad-CAM tumor localization.</p>
</div>
""", unsafe_allow_html=True)

if selected_image is None:
    st.info("👈 Please select a pre-loaded clinical sample or upload an MRI scan from the sidebar to begin analysis.")
else:
    # Prepare Image Array
    img_rgb = np.array(selected_image.convert("RGB"))
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_tensor = np.expand_dims(img_resized.astype("float32"), axis=0)

    # Run Prediction & Grad-CAM (using optimal fixed overlay alpha = 0.45)
    with st.spinner("Analyzing cranial MRI scan..."):
        heatmap, probs, pred_idx = compute_gradcam(img_tensor, model, base_model)
        superimposed_img, jet_heatmap = create_superimposed_overlay(img_resized, heatmap, alpha=0.45)

    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100
    is_healthy = (pred_class == "notumor")

    # Diagnostic Summary Banner
    badge_type = "badge-healthy" if is_healthy else "badge-tumor"
    status_icon = "✅" if is_healthy else "⚠️"
    status_color = "#10b981" if is_healthy else "#ef4444"

    st.markdown(f"""
    <div class="result-badge {badge_type}">
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {status_color}; margin-bottom: 4px;">
                {status_icon} Primary Diagnosis Result
            </div>
            <div class="result-title" style="color: #ffffff;">
                {CLASS_DISPLAY[pred_class]}
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">Confidence</div>
            <div class="result-confidence" style="color: {status_color};">
                {confidence:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Two Main Columns: Visual Saliency Studio (Left) & Probabilities (Right)
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # --------------------------------------------------------------------------
    # COLUMN 1: VISUAL DIAGNOSTIC STUDIO (Original vs. Grad-CAM)
    # --------------------------------------------------------------------------
    with col_left:
        st.markdown("#### 🔬 Visual Saliency (Grad-CAM)")
        
        tab_overlay, tab_heatmap, tab_side_by_side = st.tabs([
            "🩺 Superimposed Grad-CAM",
            "🔥 Pure Heatmap",
            "↔️ Side-by-Side Comparison"
        ])

        with tab_overlay:
            st.image(
                superimposed_img,
                caption="Grad-CAM Attention Overlay",
                use_container_width=True
            )
            st.caption("🔴 Red / Warm regions highlight the specific tissue driving the model's classification.")

        with tab_heatmap:
            st.image(
                jet_heatmap,
                caption="Extracted Activation Heatmap (Last Conv Layer)",
                use_container_width=True
            )

        with tab_side_by_side:
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.image(img_resized, caption="Original MRI (224×224)", use_container_width=True)
            with subcol2:
                st.image(superimposed_img, caption="Grad-CAM Overlay", use_container_width=True)

    # --------------------------------------------------------------------------
    # COLUMN 2: PROBABILITY DISTRIBUTION
    # --------------------------------------------------------------------------
    with col_right:
        st.markdown("#### 📊 Multi-Class Probabilities")
        
        # Sort classes by descending probability
        sorted_indices = np.argsort(probs)[::-1]
        
        for idx in sorted_indices:
            c_name = CLASS_NAMES[idx]
            c_prob = probs[idx] * 100
            c_color = CLASS_DESCRIPTIONS[c_name]["badge_color"]
            
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-header">
                    <span>{CLASS_DISPLAY[c_name]}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: {c_color};">{c_prob:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {c_prob:.1f}%; background-color: {c_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if actual_label is not None:
            match = (actual_label == pred_class)
            match_color = "#10b981" if match else "#f59e0b"
            match_text = "MATCHED" if match else "MISCLASSIFIED"
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="padding: 12px 16px; border-radius: 10px; background: rgba(15, 23, 42, 0.6); border: 1px solid {match_color}; font-size: 0.88rem;">
                <strong>Validation:</strong> Ground Truth: <code>{actual_label}</code> | Prediction: <code>{pred_class}</code> 
                <span style="color: {match_color}; font-weight: 700; margin-left: 6px;">[{match_text}]</span>
            </div>
            """, unsafe_allow_html=True)
