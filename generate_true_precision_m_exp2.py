#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "/scratch/gpfs/ALAINK/Suthi"
CLASS_NAMES = [
    "Boulevard", "Building shadow", "Bus_or_truck", "Car", "Crosswalk",
    "Person", "Protected bike lane", "Sidewalk", "Train tracks",
    "Tree", "Unprotected bike lane",
]
LABELS_FULL = CLASS_NAMES + ["background"]

npy_path = os.path.join(BASE, "aggregated_confusion_matrices", "yolo", "cm_agg_yolov8m_exp2.npy")
cm_full = np.load(npy_path).astype(float)
nc = len(CLASS_NAMES)
label = "yolov8m_exp2"
folds_used = 5

cm_obj_rows = cm_full[:nc, :].copy()
row_sums = cm_obj_rows.sum(axis=1, keepdims=True)
cm_prec = np.zeros_like(cm_obj_rows)
np.divide(cm_obj_rows, row_sums, out=cm_prec, where=row_sums > 0)
cm_prec[row_sums.squeeze() == 0, :] = np.nan
cm_prec[cm_prec < 0.005] = np.nan

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(
    cm_prec, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=LABELS_FULL, yticklabels=CLASS_NAMES,
    ax=ax, vmin=0, vmax=1, linewidths=0.3,
)
ax.set_title(
    f"{label} — True Precision (row-norm full matrix, n={folds_used} folds)\n"
    "Diagonal = TP/(TP+FP); last column = spurious fire rate (FP over background)",
    fontsize=10,
)
ax.set_xlabel("Actual class (including background)")
ax.set_ylabel("Predicted class")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

out_dir = os.path.join(BASE, "testing_increasing_accuracy_yolo", "true_precision_cms")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"true_precision_{label}.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
