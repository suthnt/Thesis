#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cityscapes 19 classes (Mask2Former semantic)
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# Default feature dirs to try
FEATURE_DIRS = [
    "/scratch/gpfs/ALAINK/Suthi/alexnet_features_1year",
    "/scratch/gpfs/ALAINK/Suthi/alexnet_multi_features_1year",
    "/scratch/gpfs/ALAINK/Suthi/mask2former_features_1year_cityscapes",
    "/scratch/gpfs/ALAINK/Suthi/mask2former_features_1year",
]
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/npz_visualizations"
LABEL_NAMES = ["safe", "dangerous"]


def load_npz(feat_dir):
    """Load train and test features."""
    train_path = os.path.join(feat_dir, "train_features.npz")
    test_path = os.path.join(feat_dir, "test_features.npz")
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        return None, None
    train = np.load(train_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    return train, test


def plot_summary(train, test, name, out_dir):
    """Print stats and save label distribution bar chart."""
    X_tr = train["features"]
    y_tr = train["labels"]
    X_te = test["features"]
    y_te = test["labels"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Train label distribution
    u_tr, c_tr = np.unique(y_tr, return_counts=True)
    axes[0].bar([LABEL_NAMES[i] for i in u_tr], c_tr, color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Train labels")
    axes[0].set_ylabel("Count")

    # Test label distribution
    u_te, c_te = np.unique(y_te, return_counts=True)
    axes[1].bar([LABEL_NAMES[i] for i in u_te], c_te, color=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Test labels")
    axes[1].set_ylabel("Count")

    plt.suptitle(f"{name}\nTrain: {X_tr.shape} | Test: {X_te.shape} | Dims: {X_tr.shape[1]}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}_summary.png")


def plot_pca_scatter(train, test, name, out_dir, max_samples=2000, seed=42):
    """PCA to 2D, scatter by label."""
    X_tr = train["features"]
    y_tr = train["labels"]
    X_te = test["features"]
    y_te = test["labels"]

    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    n = len(X)

    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_samples, replace=False)
        X, y = X[idx], y[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in [0, 1]:
        mask = y == lbl
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c="#2ecc71" if lbl == 0 else "#e74c3c",
            label=LABEL_NAMES[lbl],
            alpha=0.5,
            s=20,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"{name} - PCA 2D")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_pca.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}_pca.png")


def plot_concept_importance(train, test, name, out_dir):
    """
    For Mask2Former Cityscapes: bar chart + CSV of which concepts (road, bike, etc.)
    are associated with safety vs danger. Correlation + classifier coefficients.
    """
    X_tr = train["features"]
    y_tr = train["labels"]
    X_te = test["features"]
    y_te = test["labels"]
    if X_tr.shape[1] != 19:
        return

    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])

    # Correlation with label (1=dangerous)
    corr = np.array([
        np.corrcoef(X_all[:, i], y_all)[0, 1] if np.isfinite(X_all[:, i]).all() else 0
        for i in range(19)
    ])
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Classifier coefficients
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te_s))
    coeffs = clf.coef_[0]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in corr]
    ax.barh(CITYSCAPES_CLASSES, corr, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Correlation with dangerous")
    ax.set_title("What's in dangerous vs safe images?\n(ground truth)")
    ax.set_xlim(-0.35, 0.35)

    ax = axes[1]
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coeffs]
    ax.barh(CITYSCAPES_CLASSES, coeffs, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Classifier coefficient\n(positive → predicts dangerous)")
    ax.set_title(f"What the model uses\n(accuracy {acc:.1%})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_concept_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}_concept_importance.png")

    # CSV with numeric values
    csv_path = os.path.join(out_dir, f"{name}_concept_importance.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "correlation_with_dangerous", "classifier_coefficient"])
        for c, r, coef in zip(CITYSCAPES_CLASSES, corr, coeffs):
            w.writerow([c, f"{r:.4f}", f"{coef:.4f}"])
    print(f"  Saved {name}_concept_importance.csv")


def plot_feature_distributions(train, test, name, out_dir, n_dims=10):
    """Show distribution of first N feature dims (for high-dim features, just a sample)."""
    X_tr = train["features"]
    X_te = test["features"]
    n_dims = min(n_dims, X_tr.shape[1])
    dims = np.linspace(0, X_tr.shape[1] - 1, n_dims, dtype=int)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    for i, d in enumerate(dims[: min(n_dims, 10)]):
        ax = axes[i]
        ax.hist(X_tr[:, d], bins=30, alpha=0.6, label="train", color="#3498db", density=True)
        ax.hist(X_te[:, d], bins=30, alpha=0.6, label="test", color="#e74c3c", density=True)
        ax.set_title(f"Dim {d}")
        ax.legend(fontsize=8)
    for j in range(i + 1, 10):
        axes[j].axis("off")
    plt.suptitle(f"{name} - Feature distributions (sample dims)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_dists.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}_dists.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize .npz feature results")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Single feature dir (e.g. alexnet_multi_features_1year). If not set, uses all found.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--pca-samples",
        type=int,
        default=2000,
        help="Max samples for PCA (default 2000)",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip PCA (faster for very large feature sets)",
    )
    args = parser.parse_args()

    dirs = [args.dir] if args.dir else FEATURE_DIRS
    dirs = [d if os.path.isabs(d) else os.path.join("/scratch/gpfs/ALAINK/Suthi", d) for d in dirs]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Visualize .npz Features ===")
    print(f"Output: {args.output_dir}\n")

    for feat_dir in dirs:
        if not os.path.isdir(feat_dir):
            continue
        name = os.path.basename(feat_dir.rstrip("/"))
        train, test = load_npz(feat_dir)
        if train is None:
            print(f"Skipping {name} (no train/test_features.npz)")
            continue

        print(f"Processing {name}...")
        plot_summary(train, test, name, args.output_dir)
        # Concept importance (road, bike, car, etc. → safety vs danger) for Cityscapes
        if name == "mask2former_features_1year_cityscapes":
            plot_concept_importance(train, test, name, args.output_dir)
        if not args.no_pca:
            plot_pca_scatter(train, test, name, args.output_dir, max_samples=args.pca_samples)
        if train["features"].shape[1] > 20:
            plot_feature_distributions(train, test, name, args.output_dir)

    print(f"\nDone. Plots in {args.output_dir}")


if __name__ == "__main__":
    main()
