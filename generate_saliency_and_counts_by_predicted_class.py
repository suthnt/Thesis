#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

BASE = "/scratch/gpfs/ALAINK/Suthi"
OUT = os.path.join(BASE, "Images_For_thesis")
SALIENCY_DIR = os.path.join(BASE, "yolo_feature_detector/saliency_gradcam_yolo")
KFOLD_DIR = os.path.join(BASE, "balanced_multiclass_1year_kfold")
DATA_DIR = os.path.join(BASE, "OrganizedDatasetMultiClassBalanced_kfold")
YOLO_DIR = os.path.join(BASE, "yolo_feature_detector")

CLASSES = ["0_safe", "1_crash", "2_crashes", "3plus_crashes"]
CLASS_LABELS = ["0 (Safe)", "1 Crash", "2 Crashes", "3+ Crashes"]
IDX_TO_LABEL = {0: "0 (Safe)", 1: "1 Crash", 2: "2 Crashes", 3: "3+ Crashes"}

ROT_RE = re.compile(r"_r(90|180|270)\.tif$")
IMG_SIZE = 224
K = 5


def strip_rotation(filename):
    return ROT_RE.sub(".tif", filename)


def get_sorted_filenames(val_dir):
    """Replicate ImageDataGenerator(shuffle=False).flow_from_directory file order:
    classes sorted alphabetically, files sorted within each class."""
    filenames = []
    for cls in sorted(os.listdir(val_dir)):
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            filenames.append(os.path.join(cls, fname))
    return filenames


def load_predictions_all_folds():
    """Load val predictions from all 5 folds with image filenames."""
    rows = []
    for fold in range(K):
        npz_path = os.path.join(KFOLD_DIR, f"unet_multi_balanced_1year_f{fold}", "val_predictions.npz")
        val_dir = os.path.join(DATA_DIR, f"fold_{fold}", "val")
        if not os.path.exists(npz_path):
            print(f"  Fold {fold}: SKIP")
            continue
        npz = np.load(npz_path)
        y_true = npz["y_true"]
        y_pred = npz["y_pred"]
        filenames = get_sorted_filenames(val_dir)
        assert len(filenames) == len(y_true), \
            f"Fold {fold}: {len(filenames)} files vs {len(y_true)} predictions"
        for i, fname in enumerate(filenames):
            full_path = os.path.join(val_dir, fname)
            rows.append({
                "image_path": full_path,
                "fold": fold,
                "true_idx": int(y_true[i]),
                "pred_idx": int(y_pred[i]),
                "true_label": IDX_TO_LABEL[int(y_true[i])],
                "pred_label": IDX_TO_LABEL[int(y_pred[i])],
                "basename": os.path.basename(fname),
                "base_chip": strip_rotation(os.path.basename(fname)),
            })
        print(f"  Fold {fold}: {len(filenames)} images")
    return pd.DataFrame(rows)


# ── 1. Load correct predictions ──
print("=== Loading predictions from all 5 folds ===")
pred_df = load_predictions_all_folds()
print(f"Total: {len(pred_df)} val images")
print("Pred distribution:", pred_df["pred_label"].value_counts().to_dict())

# ── 2. Saliency heatmap by PREDICTED class ──
print("\n=== Saliency heatmap by predicted class ===")
sal_csv = os.path.join(SALIENCY_DIR, "outputs_multiclass_gradcam_fold0/saliency_yolo_detections_m.csv")
sal_df = pd.read_csv(sal_csv)
sal_df["mean_saliency"] = pd.to_numeric(sal_df["mean_saliency"], errors="coerce")
sal_df = sal_df.dropna(subset=["mean_saliency"])

# Map image_path to correct pred_label from val_predictions
fold0_preds = pred_df[pred_df["fold"] == 0][["image_path", "pred_label", "pred_idx"]].copy()
# Join saliency detections to correct predictions
sal_df = sal_df.merge(fold0_preds[["image_path", "pred_label"]], on="image_path", how="left")
n_matched = sal_df["pred_label"].notna().sum()
print(f"  Matched {n_matched}/{len(sal_df)} detections to correct predictions")
sal_df = sal_df.dropna(subset=["pred_label"])

# Pivot: predicted class × YOLO feature → mean saliency
pivot = sal_df.pivot_table(index="pred_label", columns="yolo_class_name",
                           values="mean_saliency", aggfunc="mean")
# Order rows
pivot = pivot.reindex([lbl for lbl in CLASS_LABELS if lbl in pivot.index])
# Top features by overall mean
overall = sal_df.groupby("yolo_class_name")["mean_saliency"].mean().sort_values(ascending=False)
top_cols = [c for c in overall.head(12).index if c in pivot.columns]
pivot_top = pivot[top_cols]

fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(top_cols)), max(4, 1.2 * len(pivot_top.index))))
im = ax.imshow(pivot_top.values.astype(float), aspect="auto", cmap="YlOrRd")
ax.set_xticks(np.arange(len(top_cols)))
ax.set_xticklabels(top_cols, rotation=45, ha="right", fontsize=9)
ax.set_yticks(np.arange(len(pivot_top.index)))
ax.set_yticklabels(pivot_top.index, fontsize=10)
ax.set_ylabel("Predicted class")
ax.set_title("Mean saliency by predicted class × feature")
plt.colorbar(im, ax=ax, label="Mean saliency")
plt.tight_layout()
heatmap_path = os.path.join(OUT, "saliency_heatmap_multiclass_by_predicted.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {heatmap_path}")

# ── 3. Saliency bar charts by predicted class (fold 0 with correct preds) ──
print("\n=== Saliency bar charts by predicted class ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, lbl in zip(axes.ravel(), CLASS_LABELS):
    sub = sal_df[sal_df["pred_label"] == lbl]
    n_img = sub["image_path"].nunique()
    n_det = len(sub)
    if n_det == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Predicted: {lbl}\n(n_images=0)", fontsize=10)
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
sal_bar_path = os.path.join(OUT, "saliency_by_predicted_class_multiclass.png")
plt.savefig(sal_bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {sal_bar_path}")

# ── 4. Mean object count by predicted class (all 5 folds) ──
print("\n=== Mean object count by predicted class (all 5 folds) ===")
yolo_csv = os.path.join(YOLO_DIR, "feature_prevalence_full_exp2_m_per_image.csv")
yolo_df = pd.read_csv(yolo_csv)
yolo_df["basename"] = yolo_df["path"].apply(lambda p: os.path.basename(p))
yolo_df = yolo_df.set_index("basename")

FEATURES = [
    "Boulevard", "Building shadow", "Bus_or_truck", "Car", "Crosswalk",
    "Person", "Protected bike lane", "Sidewalk", "Train tracks", "Tree",
    "Unprotected bike lane",
]

joined = pred_df.merge(yolo_df[FEATURES], left_on="base_chip", right_index=True, how="left")
n_matched = joined[FEATURES[0]].notna().sum()
print(f"  Matched to YOLO: {n_matched}/{len(joined)}")
joined = joined.dropna(subset=FEATURES)

grouped = joined.groupby("pred_label")[FEATURES].mean()
grouped = grouped.reindex([lbl for lbl in CLASS_LABELS if lbl in grouped.index])

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
y_max = grouped.max().max() * 1.15

for i, (lbl, color) in enumerate(zip(CLASS_LABELS, colors)):
    if lbl not in grouped.index:
        continue
    vals = grouped.loc[lbl].values
    n_images = int((joined["pred_label"] == lbl).sum())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(FEATURES))
    bars = ax.bar(x, vals, color=color, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0.05:
            ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Mean Object Count per Image", fontsize=12)
    ax.set_title(f"Predicted: {lbl}\nMean YOLO Object Count (n={n_images} images, 5-Fold CV)", fontsize=13)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    safe_name = lbl.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
    plot_path = os.path.join(OUT, f"unet_multiclass_yolo_counts_pred_{safe_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

print("\nDone!")
