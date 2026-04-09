#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "/scratch/gpfs/ALAINK/Suthi"
RUNS = os.path.join(BASE, "equally_balanced_multiclass_runs")
DST = os.path.join(BASE, "Images_For_thesis")
os.makedirs(DST, exist_ok=True)

CLASSES = ["0_safe", "1_crash", "2_crashes", "3plus_crashes"]
K = 5

MODELS = [
    ("alexnet",   "AlexNet Multiclass Confusion Matrix",   "alexnet_multiclass_cm_normalized.png"),
    ("firstcnn",  "FirstCNN Multiclass Confusion Matrix",  "firstcnn_multiclass_cm_normalized.png"),
    ("resnet50",  "ResNet50 Multiclass Confusion Matrix",  "resnet50_multiclass_cm_normalized.png"),
]


def main():
    for model_dir, title, out_name in MODELS:
        cm_agg = np.zeros((len(CLASSES), len(CLASSES)))
        folds_used = 0

        for fold in range(K):
            npy = os.path.join(RUNS, model_dir, f"fold_{fold}", "confusion_matrix.npy")
            if not os.path.exists(npy):
                print(f"  {model_dir} fold {fold}: SKIP (no .npy)")
                continue
            cm_agg += np.load(npy)
            folds_used += 1

        if folds_used < K:
            print(f"  {model_dir}: only {folds_used}/{K} folds — skipping")
            continue

        # Row-normalize
        row_sums = cm_agg.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_agg, row_sums, where=row_sums > 0)

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                    vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        out_path = os.path.join(DST, out_name)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    print(f"\nDone — outputs in {DST}")


if __name__ == "__main__":
    main()
