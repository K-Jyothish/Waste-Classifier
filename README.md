
# ♻️ Waste Material Classifier (CLIP-based)

A **Gradio web application** that classifies waste material images using **CLIP embeddings**, comparing **zero-shot**, **few-shot**, and **unsupervised** learning approaches.



---

## 🚀 Features

* **Zero-Shot CLIP Classification** (text–image similarity, no training)
* **Few-Shot SVM** trained on CLIP embeddings
* **Unsupervised KMeans Clustering** for visual grouping
* **Explainability Heatmap** (occlusion-based)
* Clean, interactive **Gradio UI**
* Robust model loading & sklearn-compatible embeddings

---

## 🧠 Approaches Used

* **Zero-Shot CLIP:** Classifies images using text prompts
* **Few-Shot SVM:** Supervised learning on CLIP features
* **KMeans Clustering:** Groups similar waste images without labels

---

## 🏗️ Project Structure

```
app/
 └── app.py                  # Main Gradio application
utils/
 └── zero_shot.py            # CLIP utilities
models/
 ├── few_shot_svm.pkl
 ├── kmeans.pkl
 └── zero_shot_config.json
outputs/
 └── clip_embeddings.npz
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
python app/app.py
```

Launches locally at:

```
http://127.0.0.1:7860
```

---

## 🔍 Explainability

Optional **occlusion heatmap** highlights image regions that influenced the zero-shot prediction.

---


