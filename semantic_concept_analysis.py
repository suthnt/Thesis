#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Cityscapes semantic segmentation - 19 classes (same order as Mask2Former output)
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# Paths
FEATURES_DIR = "/scratch/gpfs/ALAINK/Suthi/mask2former_features_1year_cityscapes"
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/semantic_concept_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load Mask2Former Cityscapes features."""
    train = np.load(os.path.join(FEATURES_DIR, "train_features.npz"), allow_pickle=True)
    test = np.load(os.path.join(FEATURES_DIR, "test_features.npz"), allow_pickle=True)
    X_train = train["features"]
    y_train = train["labels"]
    f_train = train["filenames"]
    X_test = test["features"]
    y_test = test["labels"]
    f_test = test["filenames"]
    return X_train, y_train, f_train, X_test, y_test, f_test


def correlation_with_label(X, y):
    """Correlation of each feature dimension with binary label (1=dangerous)."""
    n = len(y)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-8] = 1.0
    corr = np.array([
        np.corrcoef(X[:, i], y)[0, 1] if np.isfinite(X[:, i]).all() else 0
        for i in range(X.shape[1])
    ])
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def classifier_coefficients(X_train, y_train, X_test, y_test):
    """Train logistic regression, return coefficients (which concepts the model uses)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    return clf.coef_[0], scaler, clf, acc


def plot_concept_importance(corr_gt, coeffs, acc, out_path):
    """Bar chart: which concepts matter for (a) ground truth, (b) classifier."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: correlation with ground truth
    ax = axes[0]
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in corr_gt]
    bars = ax.barh(CITYSCAPES_CLASSES, corr_gt, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Correlation with dangerous (1) vs safe (0)")
    ax.set_title("What's in dangerous vs safe images?\n(Ground truth labels)")
    ax.set_xlim(-0.35, 0.35)

    # Right: classifier coefficients
    ax = axes[1]
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coeffs]
    bars = ax.barh(CITYSCAPES_CLASSES, coeffs, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Logistic regression coefficient\n(positive = predicts dangerous)")
    ax.set_title(f"What the classifier uses\n(accuracy {acc:.1%})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def find_similar_images(
    example_paths,
    top_k=20,
    output_subdir="similar_to_concept",
):
    """
    Given example image path(s), extract their Mask2Former features (from our precomputed
    features - we need to match by filename). So example_paths should be filenames or
    full paths that we can match to f_test/f_train.

    Actually: we need to get features for arbitrary images. The user provides paths.
    Our precomputed features are keyed by filename. So we need to:
    - If user gives paths that exist in our test set (by basename), use those
    - Otherwise we'd need to run Mask2Former on new images - more complex.

    Simpler: user provides a list of *filenames* or paths. We find which of our
    test/train samples match (by basename). We average their feature vectors to get
    the "concept". Then we score all test images by cosine similarity to that concept.
    Output: ranked list of (filename, score, label) and optionally copy images to a folder.

    Let's implement: example_paths can be full paths or basenames. We'll search
    f_train and f_test for matches.
    """
    # Build index: filename -> (features, label, split)
    X_train, y_train, f_train, X_test, y_test, f_test = load_data()
    all_X = np.vstack([X_train, X_test])
    all_y = np.concatenate([y_train, y_test])
    all_f = list(f_train) + list(f_test)

    name_to_idx = {}
    for i, fn in enumerate(all_f):
        base = os.path.basename(str(fn))
        name_to_idx[base] = i

    # Get concept vector from examples
    concept_indices = []
    for p in example_paths:
        base = os.path.basename(p)
        if base in name_to_idx:
            concept_indices.append(name_to_idx[base])
        else:
            print(f"  Warning: {base} not found in features (skip)")
    if not concept_indices:
        raise ValueError(
            "None of the example images were found in the feature set. "
            "Use filenames from OrganizedDatasetBalancedBinary train/test."
        )

    concept_vec = all_X[concept_indices].mean(axis=0)
    concept_vec = concept_vec / (np.linalg.norm(concept_vec) + 1e-8)

    # Score ALL images (train + test) by cosine similarity
    X_all_norm = all_X / (np.linalg.norm(all_X, axis=1, keepdims=True) + 1e-8)
    scores = X_all_norm @ concept_vec

    # Exclude the example images from results
    scores[concept_indices] = -np.inf
    order = np.argsort(scores)[::-1]
    top_indices = order[:top_k]

    results = []
    for idx in top_indices:
        fn = str(all_f[idx])
        score = scores[idx]
        label = all_y[idx]
        results.append((fn, score, label))

    # Save results to CSV
    out_csv = os.path.join(OUTPUT_DIR, f"{output_subdir}_ranked.csv")
    os.makedirs(os.path.dirname(out_csv) or OUTPUT_DIR, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("rank,filename,similarity_score,label\n")
        for r, (fn, sc, lbl) in enumerate(results, 1):
            f.write(f"{r},{fn},{sc:.4f},{lbl}\n")
    print(f"Saved ranked list: {out_csv}")

    # Copy top images to folder for easy viewing
    out_img_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(out_img_dir, exist_ok=True)
    for r, (fn, sc, lbl) in enumerate(results[:top_k], 1):
        for split in ["train", "test"]:
            for label_name in ["safe", "dangerous"]:
                src = os.path.join(DATASET_DIR, split, label_name, fn)
                if os.path.isfile(src):
                    dest = os.path.join(out_img_dir, f"{r:02d}_score{sc:.3f}_{fn}")
                    shutil.copy2(src, dest)
                    break
            else:
                continue
            break

    print(f"Copied top {top_k} images to: {out_img_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Semantic concept analysis: quantify and retrieve by concept."
    )
    parser.add_argument(
        "--retrieve",
        nargs="+",
        metavar="IMAGE",
        help="Find images similar to these example(s). Give paths or filenames.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of similar images to return (default: 20)",
    )
    parser.add_argument(
        "--concept-name",
        type=str,
        default="concept",
        help="Name for output folder (e.g. bike_lane)",
    )
    args = parser.parse_args()

    print("=== Semantic Concept Analysis ===")
    X_train, y_train, f_train, X_test, y_test, f_test = load_data()
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Concepts: {len(CITYSCAPES_CLASSES)} Cityscapes classes\n")

    # 1. Correlation with ground truth
    corr_gt = correlation_with_label(np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]))
    print("Correlation with 'dangerous' (ground truth):")
    for c, r in sorted(zip(CITYSCAPES_CLASSES, corr_gt), key=lambda x: -abs(x[1])):
        print(f"  {c:15s}: {r:+.3f}")
    print()

    # 2. Classifier coefficients
    coeffs, scaler, clf, acc = classifier_coefficients(X_train, y_train, X_test, y_test)
    print("Classifier coefficients (what the model uses):")
    for c, w in sorted(zip(CITYSCAPES_CLASSES, coeffs), key=lambda x: -abs(x[1])):
        print(f"  {c:15s}: {w:+.3f}")
    print()

    # 3. Plot
    plot_concept_importance(
        corr_gt,
        coeffs,
        acc,
        os.path.join(OUTPUT_DIR, "concept_importance.png"),
    )

    # 4. Optional: concept-based retrieval
    if args.retrieve:
        print(f"\n=== Concept retrieval: similar to {args.retrieve} ===")
        find_similar_images(
            args.retrieve,
            top_k=args.top,
            output_subdir=args.concept_name,
        )

    print("\n=== Done ===")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
