# This code was written with the assistance of Claude (Anthropic).

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# === CONFIG: Add feature extraction directories to combine ===
# Combine multiple feature sources - add paths as they become available
FEATURE_DIRS = [
    "/scratch/gpfs/ALAINK/Suthi/mask2former_features_1year_cityscapes",  # 19 dims
    "/scratch/gpfs/ALAINK/Suthi/mask2former_features_1year",             # 133 dims (COCO)
    "/scratch/gpfs/ALAINK/Suthi/alexnet_features_1year",                 # 4096 dims (binary)
    "/scratch/gpfs/ALAINK/Suthi/alexnet_multi_features_1year",            # 4096 dims (multi-class)
    "/scratch/gpfs/ALAINK/Suthi/unet_features_1year",                     # 256 dims (U-Net binary)
]
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/meta_classifier_output"
CLASS_NAMES = ["safe", "dangerous"]
CLASSIFIER = "logistic"  # "logistic" or "rf"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_align_features(feature_dirs):
    """Load features from multiple dirs, align by filename, return concatenated arrays."""
    train_features_list = []
    test_features_list = []
    train_labels = None
    test_labels = None
    train_filenames = None
    test_filenames = None

    for i, feat_dir in enumerate(feature_dirs):
        train_path = os.path.join(feat_dir, "train_features.npz")
        test_path = os.path.join(feat_dir, "test_features.npz")

        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            print(f"  Skipping {feat_dir} (files not found)")
            continue

        data_train = np.load(train_path, allow_pickle=True)
        data_test = np.load(test_path, allow_pickle=True)

        X_tr = data_train["features"]
        X_te = data_test["features"]
        y_tr = data_train["labels"]
        y_te = data_test["labels"]
        f_tr = data_train.get("filenames", np.arange(len(X_tr)))
        f_te = data_test.get("filenames", np.arange(len(X_te)))

        if train_filenames is None:
            train_filenames = f_tr
            test_filenames = f_te
            train_labels = y_tr
            test_labels = y_te
        else:
            if len(X_tr) != len(train_filenames) or len(X_te) != len(test_filenames):
                print(f"  WARNING: {feat_dir} has different sample counts - aligning by index (may be wrong)")
                min_tr = min(len(X_tr), len(train_filenames))
                min_te = min(len(X_te), len(test_filenames))
                X_tr, train_filenames, train_labels = X_tr[:min_tr], train_filenames[:min_tr], train_labels[:min_tr]
                X_te, test_filenames, test_labels = X_te[:min_te], test_filenames[:min_te], test_labels[:min_te]

        train_features_list.append(X_tr)
        test_features_list.append(X_te)
        print(f"  + {os.path.basename(feat_dir)}: train {X_tr.shape}, test {X_te.shape}")

    if not train_features_list:
        raise RuntimeError("No feature files found. Add valid paths to FEATURE_DIRS.")

    X_train = np.hstack(train_features_list)
    X_test = np.hstack(test_features_list)
    return X_train, X_test, train_labels, test_labels, train_filenames, test_filenames


def main():
    print("=== Meta-Classifier (combined features) ===")
    print("Feature sources:")
    X_train, X_test, y_train, y_test, _, _ = load_and_align_features(FEATURE_DIRS)

    print(f"\nCombined: train {X_train.shape}, test {X_test.shape}")

    # Scale features (different sources have different ranges)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    if CLASSIFIER == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        clf = LogisticRegression(max_iter=2000, random_state=42)

    print(f"\nTraining {CLASSIFIER}...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Plots
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"Meta-Classifier ({CLASSIFIER})\nCombined Features")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_meta.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/confusion_matrix_meta.png")

    # Metrics bar chart
    metrics = {
        "Accuracy": accuracy,
        "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    bars = ax.bar(metrics.keys(), list(metrics.values()), color=colors[: len(metrics)])
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_title("Meta-Classifier Performance")
    ax.set_ylim(0, 1.15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_metrics_meta.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/accuracy_metrics_meta.png")

    print(f"\n=== Done ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
