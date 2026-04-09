#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE = "/scratch/gpfs/ALAINK/Suthi"
BASE_KFOLD = os.path.join(BASE, "balanced_multiclass_1year_kfold")
BASE_DATA = os.path.join(BASE, "OrganizedDatasetMultiClassBalanced_kfold")
YOLO_DIR = os.path.join(BASE, "yolo_feature_detector")
OUT_DIR = os.path.join(BASE, "Images_For_thesis")
K = 5
CLASSES = ["0_safe", "1_crash", "2_crashes", "3plus_crashes"]
CLASS_LABELS = ["0 (Safe)", "1 Crash", "2 Crashes", "3+ Crashes"]
FEATURES = [
    "Boulevard", "Building shadow", "Bus_or_truck", "Car", "Crosswalk",
    "Person", "Protected bike lane", "Sidewalk", "Train tracks", "Tree",
    "Unprotected bike lane",
]
IMG_SIZE = 224  # U-Net

# Rotation suffix pattern
ROT_RE = re.compile(r"_r(90|180|270)\.tif$")


def strip_rotation(filename):
    """chip_000801_40.876861_-73.859752_r90.tif -> chip_000801_40.876861_-73.859752.tif"""
    return ROT_RE.sub(".tif", filename)


def main():
    # Load YOLO per-image counts (keyed by basename)
    yolo_csv = os.path.join(YOLO_DIR, "feature_prevalence_full_exp2_m_per_image.csv")
    print(f"Loading YOLO counts from {yolo_csv}")
    yolo_df = pd.read_csv(yolo_csv)
    yolo_df["basename"] = yolo_df["path"].apply(lambda p: os.path.basename(p))
    yolo_df = yolo_df.set_index("basename")

    # Collect all val images with their U-Net predictions across folds
    rows = []
    for fold in range(K):
        preds_path = os.path.join(BASE_KFOLD, f"unet_multi_balanced_1year_f{fold}", "val_predictions.npz")
        val_dir = os.path.join(BASE_DATA, f"fold_{fold}", "val")

        if not os.path.exists(preds_path):
            print(f"  Fold {fold}: SKIP (no predictions)")
            continue

        npz = np.load(preds_path)
        y_true = npz["y_true"]
        y_pred = npz["y_pred"]

        # Rebuild file order using the same generator settings
        gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
            val_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=32,
            class_mode="categorical", shuffle=False,
        )
        filenames = gen.filenames  # e.g. "0_safe/chip_xxx.tif"

        assert len(filenames) == len(y_true), f"Fold {fold}: {len(filenames)} files vs {len(y_true)} predictions"

        for i, fname in enumerate(filenames):
            basename = os.path.basename(fname)
            base_chip = strip_rotation(basename)
            rows.append({
                "fold": fold,
                "filename": basename,
                "base_chip": base_chip,
                "true_class_idx": int(y_true[i]),
                "pred_class_idx": int(y_pred[i]),
                "true_class": CLASSES[int(y_true[i])],
                "pred_class": CLASSES[int(y_pred[i])],
            })

        print(f"  Fold {fold}: {len(filenames)} images")

    pred_df = pd.DataFrame(rows)
    print(f"\nTotal val images across folds: {len(pred_df)}")

    # Join with YOLO counts on base_chip
    joined = pred_df.merge(
        yolo_df[FEATURES], left_on="base_chip", right_index=True, how="left"
    )
    n_matched = joined[FEATURES[0]].notna().sum()
    n_missing = joined[FEATURES[0]].isna().sum()
    print(f"Matched to YOLO: {n_matched}, Missing: {n_missing}")

    # Drop unmatched (rotated copies of chips not in full dataset shouldn't happen,
    # but some kfold val chips may come from the balanced set, not original)
    joined = joined.dropna(subset=FEATURES)

    # Group by predicted class → mean counts per feature
    grouped = joined.groupby("pred_class")[FEATURES].mean()
    grouped = grouped.reindex(CLASSES)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "unet_multiclass_yolo_object_counts_by_predicted_class.csv")
    grouped.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")
    print(grouped.round(3).to_string())

    # Plot: one bar chart per predicted class
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    # Compute global y-max for consistent axes
    y_max = grouped.max().max() * 1.15

    for i, (cls, label) in enumerate(zip(CLASSES, CLASS_LABELS)):
        if cls not in grouped.index:
            continue
        vals = grouped.loc[cls].values
        n_images = int((joined["pred_class"] == cls).sum())

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(FEATURES))
        bars = ax.bar(x, vals, color=colors[i], alpha=0.85, edgecolor="white")

        # Add value labels
        for bar, v in zip(bars, vals):
            if v > 0.005:
                ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=11)
        ax.set_ylabel("Mean Object Count per Image", fontsize=12)
        ax.set_title(f"Predicted: {label}\nMean YOLO Object Counts (n={n_images} images, 5-Fold CV)", fontsize=13)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(OUT_DIR, f"unet_multiclass_yolo_counts_pred_{cls}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")

    # Also compute counts per true class for reference
    grouped_true = joined.groupby("true_class")[FEATURES].mean()
    grouped_true = grouped_true.reindex(CLASSES)
    csv_true_path = os.path.join(OUT_DIR, "unet_multiclass_yolo_object_counts_by_true_class.csv")
    grouped_true.to_csv(csv_true_path)
    print(f"Saved: {csv_true_path}")


if __name__ == "__main__":
    main()
