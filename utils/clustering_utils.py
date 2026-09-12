# utils/clustering_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def cluster_embeddings(embedding_file="outputs/clip_embeddings.npz", n_clusters=5):
    """
    Perform unsupervised clustering on CLIP image embeddings and
    compare discovered clusters to the true labels.
    """
    # 1. Load saved CLIP embeddings
    data = np.load(embedding_file, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]

    # 2. Apply K-Means clustering
    print(f"🔍 Running K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    print("✅ Clustering complete!")

    # 3. Build cross-tabulation of clusters vs true labels
    cluster_vs_class = pd.crosstab(index=cluster_labels, columns=labels)
    print("\n📊 Cluster vs True Class Distribution:")
    print(cluster_vs_class)

    # 4. Plot and save heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_vs_class, annot=True, fmt="d", cmap="Blues")
    plt.title("K-Means Cluster vs True Class")
    plt.xlabel("True Label")
    plt.ylabel("Cluster ID")
    plt.tight_layout()
    plt.savefig("outputs/cluster_vs_class.png")
    plt.close()
    print("✅ Heatmap saved to outputs/cluster_vs_class.png")

    # 5. Return clustering results
    return {
        "cluster_labels": cluster_labels,
        "cluster_vs_class": cluster_vs_class
    }

if __name__ == "__main__":
    cluster_embeddings()


    # ----------------------------------------------------------
# Optional: Visualize CLIP embeddings in 2D (PCA or t-SNE)
# ----------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_clusters(embedding_file="outputs/clip_embeddings.npz", n_components=2, use_tsne=False):
    """
    Visualize CLIP embeddings in 2D using PCA or t-SNE.
    Each point represents one image, colored by its true class label.
    """
    print("🔍 Reducing dimensions for visualization...")
    data = np.load(embedding_file, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]

    if use_tsne:
        print("🌀 Using t-SNE (this may take a few minutes)...")
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
    else:
        print("⚡ Using PCA (fast mode)...")
        reducer = PCA(n_components=n_components)

    reduced = reducer.fit_transform(features)

    # Plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=labels, palette="tab10", s=25)
    plt.title("2D Visualization of CLIP Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="True Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/embedding_visualization.png")
    plt.close()
    print("✅ Embedding visualization saved to outputs/embedding_visualization.png")

