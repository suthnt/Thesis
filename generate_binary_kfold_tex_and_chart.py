#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/scratch/gpfs/ALAINK/Suthi/Images_For_thesis"
CSV_PATH = os.path.join(OUT_DIR, "binary_cnn_accuracy_metrics.csv")

# Nicer display names
DISPLAY = {
    "FirstCNN_bin": "FirstCNN",
    "AlexNet_bin": "AlexNet",
    "ResNet50_bin": "ResNet50",
    "VGG16_1year": "VGG16",
    "InceptionV3_1year": "InceptionV3",
    "unet_1year": "U-Net",
}


def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    # ---------- LaTeX table ----------
    tex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Binary Classification Metrics Aggregated Over 5-Fold Cross-Validation}",
        r"\label{tab:binary_kfold_metrics}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Accuracy & Precision & Recall & F1 Score \\",
        r"\midrule",
    ]
    for r in rows:
        name = DISPLAY.get(r["Model"], r["Model"])
        tex_lines.append(
            f"  {name} & {float(r['Accuracy']):.4f} & {float(r['Precision']):.4f} "
            f"& {float(r['Recall']):.4f} & {float(r['F1']):.4f} \\\\"
        )
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = os.path.join(OUT_DIR, "binary_kfold_metrics.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"LaTeX table -> {tex_path}")

    # ---------- Bar chart ----------
    names = [DISPLAY.get(r["Model"], r["Model"]) for r in rows]
    acc = [float(r["Accuracy"]) for r in rows]
    f1 = [float(r["F1"]) for r in rows]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_acc = ax.bar(x - width / 2, acc, width, label="Accuracy", color="#3498db")
    bars_f1 = ax.bar(x + width / 2, f1, width, label="Macro F1", color="#e74c3c")

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Binary Classification: Accuracy & F1 (5-Fold Aggregated)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars_acc:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, "binary_kfold_accuracy_f1_chart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Bar chart  -> {chart_path}")


if __name__ == "__main__":
    main()
