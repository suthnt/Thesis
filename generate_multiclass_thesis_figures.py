#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_KFOLD = "/scratch/gpfs/ALAINK/Suthi/balanced_multiclass_1year_kfold"
OUT_DIR = "/scratch/gpfs/ALAINK/Suthi/Images_For_thesis"
CLASSES = ["0_safe", "1_crash", "2_crashes", "3plus_crashes"]
CLASS_LABELS = ["0 (Safe)", "1 Crash", "2 Crashes", "3+ Crashes"]
K = 5

MODELS = [
    ("FirstCNN_multi_balanced_1year", "FirstCNN", "Our CNN's"),
    ("AlexNet_multi_balanced_1year", "AlexNet", "AlexNet"),
    ("InceptionV3_multi_balanced_1year", "InceptionV3", "InceptionV3"),
    ("VGG16_multi_balanced_1year", "VGG16", "VGG16"),
    ("unet_multi_balanced_1year", "U-Net", "U-Net"),
    ("ResNet50_multi_balanced_1year", "ResNet50", "ResNet50"),
]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load metrics from CSV ──
metrics_path = os.path.join(OUT_DIR, "multiclass_cnn_accuracy_metrics.csv")
metrics = {}
with open(metrics_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        metrics[row["model"]] = row

# ── 1. Comparison bar chart ──
print("=== Generating comparison bar chart ===")
model_names = [label for _, label, _ in MODELS]
accuracies = [float(metrics[m]["mean_val_accuracy"]) for m in model_names]
f1_macros = [float(metrics[m]["mean_f1_macro"]) for m in model_names]

x = np.arange(len(model_names))
width = 0.3

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#4394E5")
bars2 = ax.bar(x + width/2, f1_macros, width, label="Macro F1", color="#E5533D")

ax.set_ylabel("Score", fontsize=13)
ax.set_title("Multiclass Classification: Accuracy & F1 (5-Fold Aggregated)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.08:
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "multiclass_cnn_comparison_chart.png")
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {chart_path}")

# ── 2. Aggregated normalized confusion matrices ──
print("\n=== Generating aggregated confusion matrices ===")
for dir_prefix, label, display_title in MODELS:
    cm_agg = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    folds_used = 0

    for fold in range(K):
        preds_path = os.path.join(BASE_KFOLD, f"{dir_prefix}_f{fold}", "val_predictions.npz")
        if not os.path.exists(preds_path):
            print(f"  {label} fold {fold}: SKIP (no predictions)")
            continue
        npz = np.load(preds_path)
        y_true = npz["y_true"]
        y_pred = npz["y_pred"]
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
        cm_agg += cm
        folds_used += 1

    if folds_used == 0:
        print(f"  {label}: no folds, skipping")
        continue

    print(f"  {label}: {folds_used} folds, {cm_agg.sum()} total samples")

    # Raw counts
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_agg, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_title(f"{display_title} Multi-Class Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    raw_path = os.path.join(OUT_DIR, f"confusion_matrix_{label}_multiclass_aggregated.png")
    plt.savefig(raw_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {raw_path}")

    # Normalized (recall per class)
    row_sums = cm_agg.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_agg.astype(float), row_sums, where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax, vmin=0, vmax=1)
    ax.set_title(f"{display_title} Multi-Class Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    norm_path = os.path.join(OUT_DIR, f"confusion_matrix_{label}_multiclass_aggregated_normalized.png")
    plt.savefig(norm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {norm_path}")

print("\nDone! All outputs in:", OUT_DIR)
