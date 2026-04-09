#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = "/scratch/gpfs/ALAINK/Suthi/Images_For_thesis"
SALIENCY = "/scratch/gpfs/ALAINK/Suthi/yolo_feature_detector/saliency_gradcam_yolo"

# ── Multiclass ──
print("=== Multiclass ===")
df = pd.read_csv(os.path.join(SALIENCY, "outputs_multiclass_gradcam_fold0/saliency_yolo_detections_m.csv"))
df["pred_class_idx_unet"] = pd.to_numeric(df["pred_class_idx_unet"], errors="coerce")
df = df.dropna(subset=["pred_class_idx_unet", "mean_saliency"])
idx_to_label = {0: "0 (Safe)", 1: "1 Crash", 2: "2 Crashes", 3: "3+ Crashes"}
df["pred_label"] = df["pred_class_idx_unet"].astype(int).map(idx_to_label)

CLASS_ORDER = ["0 (Safe)", "1 Crash", "2 Crashes", "3+ Crashes"]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, lbl in zip(axes.ravel(), CLASS_ORDER):
    sub = df[df["pred_label"] == lbl]
    n_img = sub["image_path"].nunique()
    n_det = len(sub)
    if n_det == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{lbl}\n(n_images=0, n_det=0)", fontsize=9)
        continue
    g = sub.groupby("yolo_class_name")["mean_saliency"].agg(["count", "mean"]).reset_index()
    g = g[g["count"] >= 5].sort_values("mean", ascending=False).head(12)
    y = np.arange(len(g))
    ax.barh(y, g["mean"].values, color="#e74c3c", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(g["yolo_class_name"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean saliency in OBB")
    ax.set_title(f"Predicted: {lbl}\n(n_images={n_img}, n_det={n_det})", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Mean Grad-CAM saliency by YOLO class — by U-Net multiclass predicted class",
             fontsize=12, y=1.02)
plt.tight_layout()
p = os.path.join(OUT, "saliency_by_predicted_class_multiclass.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ── Binary ──
print("\n=== Binary ===")
df2 = pd.read_csv(os.path.join(SALIENCY, "outputs_binary_gradcam_fold0/saliency_yolo_detections_m.csv"))
df2["pred_class_idx_unet"] = pd.to_numeric(df2["pred_class_idx_unet"], errors="coerce")
df2 = df2.dropna(subset=["pred_class_idx_unet", "mean_saliency"])
idx_to_bin = {0: "dangerous", 1: "safe"}
df2["pred_label"] = df2["pred_class_idx_unet"].astype(int).map(idx_to_bin)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, lbl in zip(axes, ["dangerous", "safe"]):
    sub = df2[df2["pred_label"] == lbl]
    n_img = sub["image_path"].nunique()
    n_det = len(sub)
    g = sub.groupby("yolo_class_name")["mean_saliency"].agg(["count", "mean"]).reset_index()
    g = g[g["count"] >= 5].sort_values("mean", ascending=False).head(12)
    y = np.arange(len(g))
    ax.barh(y, g["mean"].values, color="#e74c3c", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(g["yolo_class_name"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean saliency in OBB")
    ax.set_title(f"Predicted: {lbl}\n(n_images={n_img}, n_det={n_det})", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Mean Grad-CAM saliency by YOLO class — by U-Net binary predicted class",
             fontsize=12, y=1.02)
plt.tight_layout()
p2 = os.path.join(OUT, "saliency_by_predicted_class_binary.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p2}")
