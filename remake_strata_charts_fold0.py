#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/scratch/gpfs/ALAINK/Suthi"
CSV_PATH = os.path.join(
    BASE, "yolo_feature_detector/saliency_gradcam_yolo/outputs_binary_gradcam_fold0",
    "saliency_yolo_detections_m.csv",
)
DST_DIR = os.path.join(BASE, "Images_For_thesis")
os.makedirs(DST_DIR, exist_ok=True)

CELLS = ("TP", "TN", "Type_I_FP", "Type_II_FN")
CELL_TITLES = {
    "TP":          "True positive\ndangerous → pred dangerous",
    "TN":          "True negative\nsafe → pred safe",
    "Type_I_FP":   "Type I (false positive)\nsafe → pred dangerous",
    "Type_II_FN":  "Type II (false negative)\ndangerous → pred safe",
}

DANGEROUS_LABEL = "dangerous"
SAFE_LABEL = "safe"
DANGEROUS_PRED = 0
SAFE_PRED = 1
TOP_N = 14
MIN_COUNT = 1


def assign_cell(row):
    y = str(row["binary_label"]).strip()
    p = int(row["pred_class_idx_unet"])
    if y == DANGEROUS_LABEL and p == DANGEROUS_PRED:
        return "TP"
    if y == SAFE_LABEL and p == SAFE_PRED:
        return "TN"
    if y == SAFE_LABEL and p == DANGEROUS_PRED:
        return "Type_I_FP"
    if y == DANGEROUS_LABEL and p == SAFE_PRED:
        return "Type_II_FN"
    return None


def aggregate_by_class(sub):
    if len(sub) == 0:
        return pd.DataFrame(columns=["yolo_class_name", "count", "mean_saliency"])
    g = sub.groupby("yolo_class_name", sort=False).agg(
        count=("mean_saliency", "count"),
        mean_saliency=("mean_saliency", "mean"),
    ).reset_index()
    return g.sort_values("count", ascending=False)


def main():
    df = pd.read_csv(CSV_PATH)
    df["pred_class_idx_unet"] = pd.to_numeric(df["pred_class_idx_unet"], errors="coerce")
    df = df.dropna(subset=["pred_class_idx_unet", "mean_saliency"])
    df["pred_class_idx_unet"] = df["pred_class_idx_unet"].astype(int)
    df["confusion_cell"] = df.apply(assign_cell, axis=1)

    # --- Chart 1: Mean saliency ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for ax, cell in zip(axes, CELLS):
        sub = df[df["confusion_cell"] == cell]
        agg = aggregate_by_class(sub)
        agg = agg[agg["count"] >= MIN_COUNT].head(TOP_N)
        if agg.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            y = np.arange(len(agg))
            ax.barh(y, agg["mean_saliency"].values, color="#e74c3c", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(agg["yolo_class_name"].values, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Mean saliency in OBB")
        ax.set_title(CELL_TITLES[cell], fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle(
        "Mean Grad-CAM saliency by YOLO class — stratified by U-Net confusion cell",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    p1 = os.path.join(DST_DIR, "confusion_strata_mean_saliency_2x2.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p1}")

    # --- Chart 2: Detection share ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for ax, cell in zip(axes, CELLS):
        sub = df[df["confusion_cell"] == cell]
        agg = aggregate_by_class(sub)
        agg = agg[agg["count"] >= MIN_COUNT].head(TOP_N)
        n_det = len(sub)
        if agg.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            y = np.arange(len(agg))
            if n_det > 0:
                vals = 100.0 * (agg["count"].values.astype(float) / n_det)
            else:
                vals = np.zeros(len(agg))
            ax.barh(y, vals, color="#3498db", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(agg["yolo_class_name"].values, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Share of detections in cell (%)")
        ax.set_title(CELL_TITLES[cell], fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle(
        "YOLO detections by class (% of all detections in cell) — stratified by U-Net confusion cell",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    p2 = os.path.join(DST_DIR, "confusion_strata_detection_share_2x2.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p2}")

    # --- Chart 3: Detection counts ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for ax, cell in zip(axes, CELLS):
        sub = df[df["confusion_cell"] == cell]
        agg = aggregate_by_class(sub)
        agg = agg[agg["count"] >= MIN_COUNT].head(TOP_N)
        if agg.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            y = np.arange(len(agg))
            ax.barh(y, agg["count"].values, color="#3498db", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(agg["yolo_class_name"].values, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Detection count")
        ax.set_title(CELL_TITLES[cell], fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle(
        "YOLO detection counts by class — stratified by U-Net confusion cell",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    p3 = os.path.join(DST_DIR, "confusion_strata_detection_counts_2x2.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p3}")


if __name__ == "__main__":
    main()
