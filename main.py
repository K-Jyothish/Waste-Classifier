# main.py
from utils.model_utils import train_few_shot_svm
from utils.zero_shot import zero_shot  # ✅ Correct import path

if __name__ == "__main__":
    # ----- Step 1: Few-Shot Model -----
    train_few_shot_svm(
        embedding_file="outputs/clip_embeddings.npz",
        model_save_path="models/few_shot_svm.pkl",
        test_size=0.2,
        perform_cv=True,
        cv_folds=5
    )

    # ----- Step 2: Improved Zero-Shot Model -----
    _ = zero_shot(
        embedding_file="outputs/clip_embeddings.npz",
        calibrate=True
    )
