import os
from tqdm import tqdm
import torch
import clip
from PIL import Image
import numpy as np

# ---------------------------------------------------------
# Load CLIP model and preprocessing
# ---------------------------------------------------------
def load_clip_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✅ CLIP model loaded on {device.upper()}")
    return model, preprocess, device

# ---------------------------------------------------------
# Extract CLIP image embeddings
# ---------------------------------------------------------
def extract_clip_embeddings(dataset_dir, output_path="outputs/clip_embeddings.npz"):
    model, preprocess, device = load_clip_model()

    image_features = []
    labels = []
    label_names = sorted(os.listdir(dataset_dir))

    print("🔍 Extracting embeddings from dataset...")

    for label_name in tqdm(label_names, desc="Processing classes"):
        class_dir = os.path.join(dataset_dir, label_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, img_name)
            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.encode_image(image)
                    features /= features.norm(dim=-1, keepdim=True)  # Normalize
                    image_features.append(features.cpu().numpy())
                    labels.append(label_name)
            except Exception as e:
                print(f"⚠️ Skipped {img_name}: {e}")

    image_features = np.concatenate(image_features, axis=0)
    labels = np.array(labels)

    # Save embeddings for reuse
    np.savez(output_path, features=image_features, labels=labels)
    print(f"✅ Saved embeddings to {output_path}")
    print(f"📊 Total images processed: {len(labels)} | Classes: {len(set(labels))}")

    return image_features, labels
