# app/app.py
"""
Waste Material Classifier (Polished & Fixed)
Developed by: Jyothish K
------------------------------------------
This app compares three approaches on waste images:
 - Zero-shot CLIP (text prompt similarity)
 - Few-shot SVM (trained on CLIP embeddings)
 - Unsupervised KMeans clustering (in CLIP space)

Fixes & improvements:
 - Ensures sklearn models receive float64 arrays (avoids buffer dtype mismatch)
 - Safe loading of models / embeddings with helpful messages
 - Clean, polished Gradio UI with optional explainability heatmap
"""

import os
import json
import io
import warnings
from typing import Optional, Tuple, List

import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt

import gradio as gr
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

# These helper functions are expected to exist in your utils module.
# load_clip: returns clip_model, preprocess_fn, device
# build_class_text_embeddings: builds text embeddings for classes (returns class_names, class_text_emb)
# PROMPT_DESCRIPTIONS: dict/list used by build_class_text_embeddings
from utils.zero_shot import (
    PROMPT_DESCRIPTIONS,
    build_class_text_embeddings,
    load_clip,
)

# ----------------------------
# Configuration
# ----------------------------
CLIP_MODEL_NAME = "ViT-B/32"
EMBEDDING_FILE = "outputs/clip_embeddings.npz"          # must contain 'features' and 'labels'
SVM_MODEL_FILE = "models/few_shot_svm.pkl"
KMEANS_MODEL_FILE = "models/kmeans.pkl"
ZERO_SHOT_CONFIG = "models/zero_shot_config.json"
N_CLUSTERS = 5

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# reduce matplotlib warnings in headless envs
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# Utilities
# ----------------------------

def ensure_numpy(x) -> np.ndarray:
    """Convert torch tensor or other types to numpy ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.array(x)

def ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Ensure numpy array is dtype float64 (required by some sklearn functions)."""
    arr = ensure_numpy(arr)
    if arr.dtype != np.float64:
        return arr.astype(np.float64)
    return arr

def safe_load_embeddings(path: str):
    """Load embeddings saved as npz with keys 'features' and 'labels'."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "features" not in data or "labels" not in data:
        raise KeyError(f"{path} must contain 'features' and 'labels' arrays")
    feats = ensure_float64(data["features"])
    labels = list(data["labels"].tolist())
    return feats, labels

def image_to_embedding(image_pil: Image.Image, clip_model, preprocess, device) -> np.ndarray:
    """
    Convert PIL image to CLIP embedding (normalized).
    Returns float64 numpy vector to be sklearn-friendly.
    """
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    image_input = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(image_input)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    vec = ensure_numpy(feat[0])
    return ensure_float64(vec)

def load_svm_model(path: str = SVM_MODEL_FILE):
    """Load Few-Shot SVM model safely (handles tuple with scaler)."""
    if os.path.exists(path):
        try:
            svm_data = joblib.load(path)
            # Fix: handle (model, scaler) tuple structure
            if isinstance(svm_data, tuple):
                model, scaler = svm_data
                return (model, scaler)
            return (svm_data, None)
        except Exception as e:
            print(f"⚠️ Failed to load SVM model at {path}: {e}")
            return None
    else:
        print(f"ℹ️ SVM model not found at {path}. Few-shot predictions will be unavailable.")
        return None

def ensure_float64(arr):
    """Ensure that array is float64 dtype (required by sklearn KMeans)."""
    if isinstance(arr, np.ndarray) and arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    return arr

def load_or_train_kmeans(path: str = KMEANS_MODEL_FILE, n_clusters: int = N_CLUSTERS):
    # If saved model exists, load. Otherwise load embeddings and fit.
    if os.path.exists(path):
        try:
            km = joblib.load(path)
            return km
        except Exception as e:
            print(f"⚠️ Failed to load KMeans model at {path}: {e} (will attempt to fit a new one)")

    # Fit from embeddings
    try:
        feats, _ = safe_load_embeddings(EMBEDDING_FILE)
    except Exception as e:
        print(f"❌ Cannot fit KMeans because embeddings couldn't be loaded: {e}")
        return None

    feats64 = ensure_float64(feats)
    print(f"🔍 Running KMeans (n_clusters={n_clusters}) on embeddings (this may take a moment)...")

    km = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    ).fit(feats64)

    try:
        joblib.dump(km, path)
        print(f"✅ Saved KMeans model to {path}")
    except Exception:
        print("⚠️ Failed to persist fitted KMeans model (not critical).")
    return km




def zero_shot_predict(image: Image.Image, clip_model, preprocess, device,
                      class_names: List[str], class_text_emb: np.ndarray, scale: float = 1.0):
    """
    Compute cosine similarities between image embedding and class text embeddings.
    class_text_emb expected to be numpy array (num_classes, dim) float64.
    Returns label, confidence(score), full_probs array.
    """
    img_vec = image_to_embedding(image, clip_model, preprocess, device)  # float64
    # class_text_emb may be torch.Tensor; convert
    class_text_emb = ensure_float64(class_text_emb)
    sims = img_vec @ class_text_emb.T  # dot product (cosine, since both normalized)
    # Numerical stabilization for softmax-like conversion
    scaled = scale * sims
    exp = np.exp(scaled - scaled.max())
    probs = exp / exp.sum()
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs

def generate_occlusion_heatmap(image: Image.Image, clip_model, preprocess, device,
                               text_vec: np.ndarray, grid: int = 12) -> Image.Image:
    """
    Occlusion sensitivity heatmap:
    - Mask patches with gray boxes and compute drop in similarity to provided text vector.
    - Returns overlay PIL image (224x224).
    Note: grid controls resolution; larger grid -> slower.
    """
    text_vec = ensure_float64(text_vec)
    image_small = image.convert("RGB").resize((224, 224))
    base_vec = image_to_embedding(image_small, clip_model, preprocess, device)
    base_score = float(np.dot(base_vec, text_vec))

    step = 224 // grid
    scores = np.zeros((grid, grid), dtype=np.float32)

    for i in range(grid):
        for j in range(grid):
            temp = image_small.copy()
            draw = ImageDraw.Draw(temp)
            x0, y0 = j * step, i * step
            draw.rectangle([x0, y0, x0 + step, y0 + step], fill=(127, 127, 127))
            masked_vec = image_to_embedding(temp, clip_model, preprocess, device)
            s = float(np.dot(masked_vec, text_vec))
            # we measure drop in similarity (how much masking reduced score)
            scores[i, j] = max(0.0, base_score - s)

    # normalize scores
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    else:
        scores = np.zeros_like(scores)

    # map scores -> colormap and upscale
    cmap = plt.get_cmap("jet")
    colored = (cmap(scores)[:, :, :3] * 255).astype(np.uint8)
    hm = Image.fromarray(colored).resize((224, 224))
    hm = ImageEnhance.Brightness(hm).enhance(1.2)

    overlay = Image.blend(image_small, hm, alpha=0.45)
    return overlay

# ----------------------------
# Load CLIP, embeddings & models
# ----------------------------
print(f"📦 Loading CLIP model ({CLIP_MODEL_NAME}) on {DEVICE}...")
try:
    clip_model, preprocess, clip_device = load_clip(device=DEVICE, model_name=CLIP_MODEL_NAME)
    # load_clip might return a 'device' separately; ensure we use consistent DEVICE string for messages
except Exception as e:
    raise RuntimeError(f"Failed to load CLIP model: {e}")

# Load embeddings (for zero-shot calibration & kmeans)
try:
    feats, labels = safe_load_embeddings(EMBEDDING_FILE)
    print(f"✅ Loaded embeddings: {feats.shape[0]} samples, dim={feats.shape[1]}")
except Exception as e:
    print(f"⚠️ Embeddings not loaded: {e}")
    feats, labels = None, None

# Zero-shot config (scale)
zero_scale = 1.0
if os.path.exists(ZERO_SHOT_CONFIG):
    try:
        with open(ZERO_SHOT_CONFIG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            zero_scale = float(cfg.get("scale", 1.0))
            print(f"🔧 Loaded zero-shot scale from config: {zero_scale}")
    except Exception:
        print("⚠️ Could not read zero-shot config; using default scale=1.0")

# Build class text embeddings (force numpy float64)
try:
    # build_class_text_embeddings might accept various signatures; the repository used it as:
    # build_class_text_embeddings(clip_model, device, prompt_descriptions=..., image_feats=feats, labels=..., weight_prompts=True)
    class_names, class_text_emb = build_class_text_embeddings(
        clip_model,
        clip_device,
        prompt_descriptions=PROMPT_DESCRIPTIONS,
        image_feats=feats,
        labels=labels,
        weight_prompts=True,
    )
    class_text_emb = ensure_float64(class_text_emb)
    class_names = [str(x) for x in class_names]
    print(f"✅ Prepared {len(class_names)} class text embeddings.")
except Exception as e:
    raise RuntimeError(f"Failed to build class text embeddings. Error: {e}")

# Load Few-shot SVM (optional)
svm_model = load_svm_model()

# Load or fit KMeans
kmeans_model = load_or_train_kmeans()

# ----------------------------
# Prediction wrapper
# ----------------------------

def classify_image(image: Image.Image, show_heatmap: bool = False) -> Tuple[str, Optional[Image.Image]]:
    """
    Main function used by the UI.
    Returns (text_summary, overlay_image_or_None)
    """
    if image is None:
        return "Please upload an image.", None

    try:
        # Zero-shot
        zs_label, zs_conf, zs_probs = zero_shot_predict(
            image, clip_model, preprocess, clip_device, class_names, class_text_emb, scale=zero_scale
        )
    except Exception as e:
        zs_label, zs_conf, zs_probs = "Error", 0.0, None
        print(f"⚠️ Zero-shot prediction failed: {e}")

    # Get image embedding (float64)
    try:
        img_vec = image_to_embedding(image, clip_model, preprocess, clip_device)  # float64
    except Exception as e:
        return f"Error extracting embedding: {e}", None

    # Few-shot SVM
    fs_label = "N/A"
    fs_conf = 0.0
    if svm_model is not None:
        try:
            model, scaler = svm_model
            img_input = img_vec.reshape(1, -1)
            if scaler is not None:
                img_input = scaler.transform(img_input)
            pred = model.predict(img_input)
            fs_label = str(pred[0])
            if hasattr(model, "predict_proba"):
                fs_conf = float(np.max(model.predict_proba(img_input)[0]))
            else:
                fs_conf = 1.0
        except NotFittedError:
            fs_label = "SVM not fitted"
        except Exception as e:
            fs_label = "Error"
            print(f"⚠️ Few-shot SVM prediction error: {e}")

    # Clustering
    cluster_info = "N/A"
    if kmeans_model is not None:
        try:
            cid = int(kmeans_model.predict(img_vec.reshape(1, -1).astype(np.float64))[0])
            cluster_info = str(cid)
        except NotFittedError:
            cluster_info = "KMeans not fitted"
        except Exception as e:
            cluster_info = "Error"
            print(f"⚠️ KMeans prediction error: {e}")




    # Compose human-friendly result text
    try:
        display_zs_label = zs_label.title() if isinstance(zs_label, str) else str(zs_label)
        display_fs_label = fs_label.title() if isinstance(fs_label, str) else str(fs_label)
    except Exception:
        display_zs_label = str(zs_label)
        display_fs_label = str(fs_label)

    result_text = (
        f"🧠 **Zero-Shot CLIP:** {display_zs_label}  —  Confidence: {zs_conf*100:.1f}%  \n"
        f"🎯 **Few-Shot SVM:** {display_fs_label}  —  Confidence: {fs_conf*100:.1f}%  \n"
        f"🔍 **Cluster Group:** {cluster_info}"
    )

    overlay = None
    if show_heatmap:
        # Use the text embedding for the predicted zero-shot class for heatmap
        try:
            if zs_label in class_names:
                idx = class_names.index(zs_label)
            else:
                # fallback to best index from zero-shot if labels differ in capitalization
                lower_map = {n.lower(): i for i, n in enumerate(class_names)}
                idx = lower_map.get(str(zs_label).lower(), 0)
            text_vec = class_text_emb[idx]
            overlay = generate_occlusion_heatmap(image, clip_model, preprocess, clip_device, text_vec, grid=12)
        except Exception as e:
            print(f"⚠️ Heatmap generation failed: {e}")
            overlay = None

    return result_text, overlay

# ----------------------------
# Gradio UI Layout
# ----------------------------
title = "♻️ Waste Material Classifier"
description = """
Upload a photo of waste material and compare three approaches:

- **Zero-Shot CLIP:** No training; uses text prompts to match images.  
- **Few-Shot SVM:** Trained on CLIP embeddings (if model exists).  
- **Clustering:** Shows the visual similarity group (KMeans).  
Enable the explainability heatmap to see which regions influenced the predicted class.
"""

# Build interface
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {background-color: #fafafa; font-family: Inter, sans-serif;}") as demo:
    gr.Markdown(f"<h1 style='text-align:center; color:#2c3e50;'>{title}</h1>")
    gr.Markdown(f"<p style='text-align:center; color:#555; font-size:15px; margin-top:-12px;'>{description}</p>")

    with gr.Row():
        with gr.Column(scale=1, min_width=420):
            img_input = gr.Image(type="pil", label="Upload Image", height=360)
            heatmap_toggle = gr.Checkbox(label="Show explainability heatmap", value=False)
            classify_btn = gr.Button("Classify Image", variant="primary")

        with gr.Column(scale=1, min_width=420):
            output_md = gr.Markdown("Upload an image and press **Classify Image**")
            explanation_img = gr.Image(label="Explanation / Heatmap (if enabled)", value=None, visible=False, height=360)

    def on_classify(image, show_heatmap):
        if image is None:
            return "Please upload an image to classify.", gr.update(visible=False, value=None)
        text, overlay = classify_image(image, show_heatmap)
        if show_heatmap and overlay is not None:
            return text, gr.update(visible=True, value=overlay)
        else:
            return text, gr.update(visible=False, value=None)

    classify_btn.click(on_classify, inputs=[img_input, heatmap_toggle], outputs=[output_md, explanation_img])

    gr.Markdown(
        "<hr><p style='text-align:center; color:#666;'>Developed by <b>Jyothish K</b> | Machine Learning Micro Project</p>"
    )

# ----------------------------
# Launch
# ----------------------------
if __name__ == "__main__":
    print("🚀 Launching Waste Classifier App (Gradio)...")
    # share=False is typical for local use; change to True if you want a temporary public link
    demo.launch(share=False, server_name="127.0.0.1")
