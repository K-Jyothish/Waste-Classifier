import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Train Few-Shot SVM with train/test split + cross-validation
# ---------------------------------------------------------
def train_few_shot_svm(
    embedding_file="outputs/clip_embeddings.npz",
    model_save_path="models/few_shot_svm.pkl",
    test_size=0.2,
    perform_cv=True,
    cv_folds=5
):
    """
    Train a few-shot SVM classifier using CLIP embeddings.

    Args:
        embedding_file: Path to saved CLIP embeddings (.npz)
        model_save_path: Where to save trained SVM + scaler
        test_size: Fraction of data to keep for testing (default 0.2)
        perform_cv: Whether to perform k-fold cross-validation
        cv_folds: Number of folds for cross-validation
    """

    print("📦 Loading CLIP embeddings...")
    data = np.load(embedding_file, allow_pickle=True)
    features, labels = data["features"], data["labels"]

    # Normalize features for better SVM training
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        stratify=labels,
        random_state=42
    )

    print(f"🧠 Training SVM on {len(X_train)} samples | Testing on {len(X_test)} samples")

    # Initialize linear SVM with regularization
    clf = svm.SVC(kernel="linear", C=0.1, probability=True, class_weight="balanced")

    # ---------------------------
    # Cross-Validation (Optional)
    # ---------------------------
    if perform_cv:
        print(f"🔁 Performing {cv_folds}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy")
        print(f"📊 Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Train on training set
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n📈 Final Test Results:")
    print(f"Accuracy :  {acc*100:.2f}%")
    print(f"Precision:  {prec*100:.2f}%")
    print(f"Recall   :  {rec*100:.2f}%")
    print(f"F1-Score :  {f1*100:.2f}%")

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=np.unique(labels),
                yticklabels=np.unique(labels))
    plt.title("Confusion Matrix - Few-Shot SVM")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/few_shot_confusion_matrix.png")
    plt.close()
    print("✅ Confusion matrix saved to outputs/few_shot_confusion_matrix.png")

    # Save trained model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump((clf, scaler), model_save_path)
    print(f"💾 SVM model and scaler saved to {model_save_path}")

    return clf, scaler, (acc, prec, rec, f1)
