#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os, re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/scratch/gpfs/ALAINK/Suthi"
OUT = os.path.join(BASE, "Images_For_thesis")
KFOLD_DIR = os.path.join(BASE, "OrganizedDatasetBalancedBinary_kfold")
YOLO_DIR = os.path.join(BASE, "yolo_feature_detector")

CLASS_LABELS = ["Dangerous", "Safe"]
IDX_TO_LABEL = {0: "Dangerous", 1: "Safe"}  # alphabetical: dangerous=0, safe=1
FOLDER_TO_IDX = {"dangerous": 0, "safe": 1}
ROT_RE = re.compile(r"_r(90|180|270)\.tif$")
K = 5

FEATURES = [
    "Boulevard", "Building shadow", "Bus_or_truck", "Car", "Crosswalk",
    "Person", "Protected bike lane", "Sidewalk", "Train tracks", "Tree",
    "Unprotected bike lane",
]


def strip_rotation(filename):
    return ROT_RE.sub(".tif", filename)


# ── 1. Check binary val_predictions.npz existence ──
has_npz = False
for fold in range(K):
    npz_candidates = [
        os.path.join(BASE, f"balanced_binary_1year_kfold/unet_bin_balanced_1year_f{fold}/val_predictions.npz"),
        os.path.join(BASE, f"cnn_binary_kfold/unet_bin_f{fold}/val_predictions.npz"),
    ]
    for p in npz_candidates:
        if os.path.exists(p):
            has_npz = True
            print(f"Found binary npz: {p}")
            break

# ── 2. Build prediction table from saliency CSV or npz ──
sal_csv = os.path.join(BASE, "yolo_feature_detector/saliency_gradcam_yolo/outputs_binary_gradcam_fold0/saliency_yolo_detections_m.csv")

if not has_npz:
    print("No binary val_predictions.npz found — using saliency CSV pred_class_idx_unet for fold 0 val images")
    sal_df = pd.read_csv(sal_csv)
    sal_val = sal_df[sal_df["split"] == "val"].copy()
    sal_val["pred_class_idx_unet"] = pd.to_numeric(sal_val["pred_class_idx_unet"], errors="coerce")
    sal_val = sal_val.dropna(subset=["pred_class_idx_unet"])

    # Get unique image → pred mapping from saliency CSV
    img_pred = sal_val.groupby("image_path")["pred_class_idx_unet"].first().astype(int).reset_index()
    img_pred.columns = ["image_path", "pred_idx"]
    img_pred["pred_label"] = img_pred["pred_idx"].map(IDX_TO_LABEL)
    img_pred["basename"] = img_pred["image_path"].apply(os.path.basename)
    img_pred["base_chip"] = img_pred["basename"].apply(strip_rotation)
    pred_df = img_pred

    # Also build for all 5 folds using kfold dataset + prediction from model must be done differently
    # Since no npz, let's use ALL folds val directories and match to saliency pred_class
    # Actually: the saliency CSV has both train and val images for fold 0 only
    # For the object count charts across 5 folds, we need predictions for all folds
    # Without npz files, we can only use fold 0 val
    print(f"  Using fold 0 val only: {len(pred_df)} unique images with predictions")
else:
    # Load from npz files (similar to multiclass script)
    rows = []
    for fold in range(K):
        val_dir = os.path.join(KFOLD_DIR, f"fold_{fold}", "val")
        for cand in npz_candidates:
            npz_path = cand.replace("f0", f"f{fold}")
            if os.path.exists(npz_path):
                break
        npz = np.load(npz_path)
        y_pred = npz["y_pred"]
        filenames = []
        for cls in sorted(os.listdir(val_dir)):
            cls_dir = os.path.join(val_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                filenames.append(os.path.join(cls, fname))
        assert len(filenames) == len(y_pred)
        for i, fname in enumerate(filenames):
            rows.append({
                "image_path": os.path.join(val_dir, fname),
                "pred_idx": int(y_pred[i]),
                "pred_label": IDX_TO_LABEL[int(y_pred[i])],
                "basename": os.path.basename(fname),
                "base_chip": strip_rotation(os.path.basename(fname)),
            })
    pred_df = pd.DataFrame(rows)

print(f"Total prediction images: {len(pred_df)}")
print("Pred distribution:", pred_df["pred_label"].value_counts().to_dict())

# ── 3. Load YOLO per-image counts ──
yolo_csv = os.path.join(YOLO_DIR, "feature_prevalence_full_exp2_m_per_image.csv")
yolo_df = pd.read_csv(yolo_csv)
yolo_df["basename"] = yolo_df["path"].apply(os.path.basename)
yolo_df = yolo_df.set_index("basename")

joined = pred_df.merge(yolo_df[FEATURES], left_on="base_chip", right_index=True, how="left")
n_matched = joined[FEATURES[0]].notna().sum()
print(f"Matched to YOLO: {n_matched}/{len(joined)}")
joined = joined.dropna(subset=FEATURES)

# ── 4. Generate individual count charts ──
grouped = joined.groupby("pred_label")[FEATURES].mean()
grouped = grouped.reindex([lbl for lbl in CLASS_LABELS if lbl in grouped.index])

colors = {"Dangerous": "#C44E52", "Safe": "#4C72B0"}
y_max = grouped.max().max() * 1.15

for lbl in CLASS_LABELS:
    if lbl not in grouped.index:
        continue
    vals = grouped.loc[lbl].values
    n_images = int((joined["pred_label"] == lbl).sum())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(FEATURES))
    bars = ax.bar(x, vals, color=colors[lbl], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0.05:
            ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Mean Object Count per Image", fontsize=12)
    ax.set_title(f"Predicted: {lbl}\nMean YOLO Object Count (n={n_images} images)", fontsize=13)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    safe_name = lbl.lower()
    plot_path = os.path.join(OUT, f"unet_binary_yolo_counts_pred_{safe_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

# ── 5. Combined 1×2 panel ──
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, lbl in zip(axes, CLASS_LABELS):
    if lbl not in grouped.index:
        continue
    vals = grouped.loc[lbl].values
    n_images = int((joined["pred_label"] == lbl).sum())
    x = np.arange(len(FEATURES))
    bars = ax.bar(x, vals, color=colors[lbl], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0.05:
            ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Object Count per Image", fontsize=11)
    ax.set_title(f"Predicted: {lbl} (n={n_images})", fontsize=12)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("Mean YOLO Object Count per Image — by U-Net Binary Predicted Class", fontsize=14, y=1.02)
plt.tight_layout()
combined_path = os.path.join(OUT, "unet_binary_yolo_counts_by_predicted_class.png")
plt.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {combined_path}")

print("\nDone!")
