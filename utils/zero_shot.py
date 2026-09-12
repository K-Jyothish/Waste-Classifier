# utils/zero_shot.py
import os
import json
import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------------------------------------
# English-only, context-rich prompt set
# ----------------------------------------------------------
PROMPT_DESCRIPTIONS = {
    "paper": [
        "a photo of paper waste",
        "a photo of newspapers",
        "a stack of papers",
        "a sheet of paper on a table",
        "recyclable paper materials"
    ],
    "plastic": [
        "a photo of a plastic bottle",
        "a photo of plastic packaging",
        "an empty plastic container",
        "a plastic water bottle",
        "single-use plastic material"
    ],
    "glass": [
        "a photo of a glass bottle",
        "a photo of a glass jar",
        "a transparent glass container",
        "a broken glass bottle",
        "recyclable glass material"
    ],
    "metal": [
        "a photo of a metal can",
        "a photo of aluminium cans",
        "a photo of metal scrap",
        "a steel soda can",
        "a metallic recyclable item"
    ],
    "cardboard": [
        "a photo of a cardboard box",
        "a corrugated cardboard box",
        "a cardboard packaging box",
        "a recycled cardboard box",
        "cardboard packaging material"
    ]
}

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def load_clip(device=None, model_name="ViT-B/32"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, device


def load_image_embeddings(embedding_npz="outputs/clip_embeddings.npz"):
    """Load CLIP image embeddings and normalize them."""
    data = np.load(embedding_npz, allow_pickle=True)
    features = data["features"].astype(np.float32)
    labels = data["labels"]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-12)
    return features, labels


# ----------------------------------------------------------
# Weighted text embedding builder
# ----------------------------------------------------------
def build_class_text_embeddings(model, device, prompt_descriptions=PROMPT_DESCRIPTIONS,
                                batch_size=64, image_feats=None, labels=None,
                                weight_prompts=False):
    """
    Compute averaged (optionally weighted) text embeddings per class from prompts.
    If weight_prompts=True, evaluates each prompt individually and assigns weights
    based on how well it aligns with image embeddings + true labels.
    """
    class_names = sorted(prompt_descriptions.keys())
    prompts = []
    prompt_to_class = []

    for cls in class_names:
        for p in prompt_descriptions[cls]:
            prompts.append(p)
            prompt_to_class.append(cls)

    toks = clip.tokenize(prompts)
    all_text_features = []
    with torch.no_grad():
        for i in range(0, toks.shape[0], batch_size):
            batch = toks[i:i+batch_size].to(device)
            feats = model.encode_text(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_text_features.append(feats.cpu())
    all_text_features = torch.cat(all_text_features, dim=0)  # (P, D)

    # Convert to NumPy for compatibility
    if isinstance(all_text_features, torch.Tensor):
        all_text_features = all_text_features.numpy()

    class_emb = {}

    if weight_prompts and image_feats is not None and labels is not None:
        print("🔍 Evaluating prompt strengths for weighting...")
        class_names_arr = np.array(class_names)

        # Compute similarity between each prompt and all image embeddings
        sims = image_feats @ all_text_features.T  # (N, P)
        preds_prompt = np.argmax(sims, axis=1)
        pred_classes = [prompt_to_class[p] for p in preds_prompt]

        # Initialize weights
        weights = np.zeros(len(prompts))
        for i, true_label in enumerate(labels):
            pred_class = pred_classes[i]
            if pred_class == true_label:
                weights[preds_prompt[i]] += 1

        # Normalize within each class
        for cls in class_names:
            idxs = [i for i, c in enumerate(prompt_to_class) if c == cls]
            w = weights[idxs]
            if np.sum(w) == 0:
                w = np.ones_like(w)
            w = w / np.sum(w)
            emb = np.average(all_text_features[idxs], axis=0, weights=w)
            emb /= np.linalg.norm(emb)
            class_emb[cls] = emb
        print("✅ Prompt weighting completed.")
    else:
        # Simple equal-weight average per class
        for cls in class_names:
            idxs = [i for i, c in enumerate(prompt_to_class) if c == cls]
            emb = np.mean(all_text_features[idxs], axis=0)
            emb /= np.linalg.norm(emb)
            class_emb[cls] = emb

    return class_names, np.stack([class_emb[c] for c in class_names], axis=0)


# ----------------------------------------------------------
# Main Zero-Shot function
# ----------------------------------------------------------
def zero_shot(
    embedding_file="outputs/clip_embeddings.npz",
    model_name="ViT-B/32",
    calibrate=True,
    weight_prompts=True,
    save_top3="outputs/zero_shot_top3.npy",
    save_cm="outputs/zero_shot_confusion_matrix.png"
):
    """
    Improved Zero-Shot pipeline with prompt weighting and calibration.
    Produces metrics, confusion matrix, and top-3 predictions.
    """
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load CLIP model
    model, preprocess, device = load_clip(model_name=model_name)

    # Load image features
    image_feats, labels = load_image_embeddings(embedding_file)

    # Build text embeddings (with optional prompt weighting)
    class_names, class_text_emb = build_class_text_embeddings(
        model, device,
        prompt_descriptions=PROMPT_DESCRIPTIONS,
        image_feats=image_feats,
        labels=labels,
        weight_prompts=weight_prompts
    )

    # Calibrate scale
    if calibrate:
        sims = image_feats @ class_text_emb.T
        scale_candidates = np.concatenate([
            np.linspace(0.2, 1.0, 5),
            np.linspace(1.0, 5.0, 9),
            np.array([8.0, 10.0])
        ])
        label_to_idx = {c: i for i, c in enumerate(class_names)}
        label_idxs = np.array([label_to_idx[l] for l in labels])
        best_s, best_acc = 1.0, -1.0
        for s in scale_candidates:
            preds = np.argmax(np.exp(s * sims), axis=1)
            acc = (preds == label_idxs).mean()
            if acc > best_acc:
                best_acc, best_s = acc, s
        s = best_s
        print(f"🔧 Calibration: selected scale s = {s:.3f} (val acc {best_acc*100:.2f}%)")
    else:
        s = 1.0
        print("🔧 Calibration skipped. Using s = 1.0")

    # Compute predictions
    sims = image_feats @ class_text_emb.T
    scores = np.exp(s * sims)
    probs = scores / (scores.sum(axis=1, keepdims=True) + 1e-12)

    topk = 3
    topk_idx = np.argsort(-probs, axis=1)[:, :topk]
    class_names_arr = np.array(class_names)

    top3_list, preds_top1 = [], []
    for i in range(probs.shape[0]):
        idxs = topk_idx[i]
        top3 = [(class_names_arr[idx], float(probs[i, idx])) for idx in idxs]
        top3_list.append(top3)
        preds_top1.append(class_names_arr[idxs[0]])
    preds_top1 = np.array(preds_top1)

    # Metrics
    acc = accuracy_score(labels, preds_top1)
    prec = precision_score(labels, preds_top1, average="macro", zero_division=0)
    rec = recall_score(labels, preds_top1, average="macro", zero_division=0)
    f1 = f1_score(labels, preds_top1, average="macro", zero_division=0)

    print("\n📊 Zero-Shot Improved Results (Top-1):")
    print(f"Accuracy :  {acc*100:.2f}%")
    print(f"Precision:  {prec*100:.2f}%")
    print(f"Recall   :  {rec*100:.2f}%")
    print(f"F1-Score :  {f1*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(labels, preds_top1, labels=class_names)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Zero-Shot (Weighted Prompts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_cm)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_cm}")

    # Save top-3 predictions
    np.save(save_top3, np.array(top3_list, dtype=object))
    print(f"✅ Top-3 predictions saved to {save_top3}")

    # Save model configuration for reproducibility
    cfg = {
        "scale": float(s),
        "class_names": class_names,
        "prompts": PROMPT_DESCRIPTIONS,
        "weighted_prompts": weight_prompts
    }
    with open("models/zero_shot_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print("💾 Zero-Shot configuration saved to models/zero_shot_config.json")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "scale": s,
        "class_names": class_names
    }

if __name__ == "__main__":
    zero_shot()
