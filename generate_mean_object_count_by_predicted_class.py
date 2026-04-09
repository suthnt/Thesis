#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = "/scratch/gpfs/ALAINK/Suthi/Images_For_thesis"
SALIENCY = "/scratch/gpfs/ALAINK/Suthi/yolo_feature_detector/saliency_gradcam_yolo"


def plot_mean_counts(df, label_col, label_order, title, out_path, ncols=2):
    nrows = int(np.ceil(len(label_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, lbl in zip(axes, label_order):
        sub = df[df[label_col] == lbl]
        n_img = sub["image_path"].nunique()
        if n_img == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Predicted: {lbl}\n(n_images=0)", fontsize=10)
            continue
        # Count detections per (image, yolo_class), then mean across images
        counts = sub.groupby(["image_path", "yolo_class_name"]).size().reset_index(name="count")
        mean_counts = counts.groupby("yolo_class_name")["count"].mean().sort_values(ascending=False).head(12)

        y = np.arange(len(mean_counts))
        ax.barh(y, mean_counts.values, color="#3498db", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(mean_counts.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Mean object count per image")
        ax.set_title(f"Predicted: {lbl}\n(n_images={n_img})", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    for j in range(len(label_order), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Multiclass ──
print("=== Multiclass ===")
df = pd.read_csv(os.path.join(SALIENCY, "outputs_multiclass_gradcam_fold0/saliency_yolo_detections_m.csv"))
df["pred_class_idx_unet"] = pd.to_numeric(df["pred_class_idx_unet"], errors="coerce")
df = df.dropna(subset=["pred_class_idx_unet"])
idx_to_label = {0: "0 (Safe)", 1: "1 Crash", 2: "2 Crashes", 3: "3+ Crashes"}
df["pred_label"] = df["pred_class_idx_unet"].astype(int).map(idx_to_label)

plot_mean_counts(
    df, "pred_label",
    ["0 (Safe)", "1 Crash", "2 Crashes", "3+ Crashes"],
    "Mean YOLO object count per image — by U-Net multiclass predicted class",
    os.path.join(OUT, "mean_object_count_by_predicted_class_multiclass.png"),
)

# ── Binary ──
print("\n=== Binary ===")
df2 = pd.read_csv(os.path.join(SALIENCY, "outputs_binary_gradcam_fold0/saliency_yolo_detections_m.csv"))
df2["pred_class_idx_unet"] = pd.to_numeric(df2["pred_class_idx_unet"], errors="coerce")
df2 = df2.dropna(subset=["pred_class_idx_unet"])
idx_to_bin = {0: "dangerous", 1: "safe"}
df2["pred_label"] = df2["pred_class_idx_unet"].astype(int).map(idx_to_bin)

plot_mean_counts(
    df2, "pred_label",
    ["dangerous", "safe"],
    "Mean YOLO object count per image — by U-Net binary predicted class",
    os.path.join(OUT, "mean_object_count_by_predicted_class_binary.png"),
    ncols=2,
)
