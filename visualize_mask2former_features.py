#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Cityscapes 19 classes (same order as Mask2Former)
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

FEATURES_DIR = "/scratch/gpfs/ALAINK/Suthi/mask2former_features_1year_cityscapes"
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/mask2former_examples"


def find_image_path(filename):
    """Locate image in train/test safe/dangerous subdirs."""
    for split in ["train", "test"]:
        for label in ["safe", "dangerous"]:
            path = os.path.join(DATASET_DIR, split, label, filename)
            if os.path.isfile(path):
                return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize images with Mask2Former feature breakdown")
    parser.add_argument("--num", type=int, default=8, help="Number of example images (default 8)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--filenames", nargs="*", help="Specific filenames to visualize")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load features
    train = np.load(os.path.join(FEATURES_DIR, "train_features.npz"), allow_pickle=True)
    test = np.load(os.path.join(FEATURES_DIR, "test_features.npz"), allow_pickle=True)
    all_features = np.vstack([train["features"], test["features"]])
    all_filenames = list(train["filenames"]) + list(test["filenames"])
    all_labels = np.concatenate([train["labels"], test["labels"]])

    name_to_idx = {str(f): i for i, f in enumerate(all_filenames)}

    if args.filenames:
        indices = []
        for fn in args.filenames:
            base = os.path.basename(fn)
            if base in name_to_idx:
                indices.append(name_to_idx[base])
            else:
                print(f"Warning: {base} not found, skipping")
        if not indices:
            print("No valid filenames found.")
            return
    else:
        # Sample randomly from test
        test_start = len(train["filenames"])
        test_indices = list(range(test_start, len(all_filenames)))
        rng = np.random.default_rng(args.seed)
        n = min(args.num, len(test_indices))
        indices = list(rng.choice(test_indices, size=n, replace=False))

    label_names = ["safe", "dangerous"]
    for idx in indices:
        feats = all_features[idx]
        fn = str(all_filenames[idx])
        lbl = all_labels[idx]
        path = find_image_path(fn)
        if path is None:
            print(f"Image not found: {fn}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        img = imread(path)
        axes[0].imshow(img)
        axes[0].set_title(f"{fn}\n({label_names[lbl]})")
        axes[0].axis("off")

        colors = ["#2ecc71" if v > 0 else "#ecf0f1" for v in feats]
        axes[1].barh(CITYSCAPES_CLASSES, feats, color=colors)
        axes[1].axvline(0, color="black", linewidth=0.5)
        axes[1].set_xlabel("Mean activation")
        axes[1].set_title("Mask2Former Cityscapes features")
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"{fn.replace('.', '_')}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    # Also create a grid of several examples
    n_show = min(6, len(indices))
    fig, axes = plt.subplots(n_show, 2, figsize=(10, 2.5 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    for i, idx in enumerate(indices[:n_show]):
        feats = all_features[idx]
        fn = str(all_filenames[idx])
        lbl = all_labels[idx]
        path = find_image_path(fn)
        if path is None:
            continue
        img = imread(path)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{label_names[lbl]}: {fn[:40]}...")
        axes[i, 0].axis("off")
        colors = ["#2ecc71" if v > 0 else "#ecf0f1" for v in feats]
        axes[i, 1].barh(CITYSCAPES_CLASSES, feats, color=colors)
        axes[i, 1].axvline(0, color="black", linewidth=0.5)
        axes[i, 1].set_xlabel("Activation")
    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, "grid_examples.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grid: {grid_path}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
