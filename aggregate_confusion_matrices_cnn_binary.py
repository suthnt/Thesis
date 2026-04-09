#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless / Slurm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

BASE = "/scratch/gpfs/ALAINK/Suthi"
K = 5
CLASSES = ["dangerous", "safe"]  # flow_from_directory sorts alphabetically

MODELS = [
    ("FirstCNN_bin_f", "FirstCNN_bin", 99, "best_FirstCNN_bin.keras"),
    ("AlexNet_bin_f", "AlexNet_bin", 227, "best_alexnet_model.keras"),
    ("ResNet50_bin_f", "ResNet50_bin", 224, "best_resnet50_model.keras"),
    ("VGG16_1year_f", "VGG16_1year", 224, "best_VGG16_1year.keras"),
    ("InceptionV3_1year_f", "InceptionV3_1year", 299, "best_InceptionV3_1year.keras"),
    ("unet_1year_f", "unet_1year", 224, "best_unet_1year.keras"),
]

# Row-normalized figures only (thesis); same order as MODELS.
THESIS_TITLES_NORMALIZED = [
    "FirstCNN binary confusion matrix",
    "AlexNet binary confusion matrix",
    "ResNet50 binary confusion matrix",
    "VGG16 binary confusion matrix",
    "InceptionV3 binary confusion matrix",
    "UNet binary confusion matrix",
]

LEGACY_OUT_DIR = os.path.join(BASE, "aggregated_confusion_matrices", "cnn_binary")
DEFAULT_THESIS_DIR = os.path.join(BASE, "Images_For_thesis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=DEFAULT_THESIS_DIR,
        help="Directory for row-normalized PNGs only (default: Images_For_thesis).",
    )
    ap.add_argument(
        "--also-save-legacy-dir",
        action="store_true",
        help="Also write raw + normalized copies under aggregated_confusion_matrices/cnn_binary/ (old behavior).",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    legacy_dir = LEGACY_OUT_DIR
    if args.also_save_legacy_dir:
        os.makedirs(legacy_dir, exist_ok=True)

    all_metrics = []  # collect per-model metrics for summary

    for i, (dir_prefix, label, img_size, ckpt_name) in enumerate(MODELS):
        cm_agg = np.zeros((2, 2))
        all_y_true = []
        all_y_pred = []
        folds_used = 0

        for fold in range(K):
            model_path = os.path.join(BASE, "binary_1year_kfold", f"{dir_prefix}{fold}", ckpt_name)
            val_dir = os.path.join(BASE, "OrganizedDatasetBalancedBinary_kfold", f"fold_{fold}", "val")

            if not os.path.exists(model_path) or not os.path.exists(val_dir):
                print(f"  {label} fold {fold}: skip (missing model or data)")
                continue

            print(f"  {label} fold {fold}: loading...")
            model = load_model(model_path)
            datagen = ImageDataGenerator(rescale=1.0 / 255)
            gen = datagen.flow_from_directory(
                val_dir,
                target_size=(img_size, img_size),
                batch_size=32,
                class_mode="categorical",
                shuffle=False,
            )
            y_pred = np.argmax(model.predict(gen), axis=1)
            y_true = gen.classes
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_agg += cm
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            folds_used += 1

        if folds_used == 0:
            print(f"  {label}: no folds found, skip")
            continue

        # Compute accuracy metrics
        y_t = np.array(all_y_true)
        y_p = np.array(all_y_pred)
        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, average="macro")
        rec = recall_score(y_t, y_p, average="macro")
        f1 = f1_score(y_t, y_p, average="macro")
        total = len(y_t)
        all_metrics.append({
            "Model": label, "Folds": folds_used, "N": total,
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        })
        print(f"  {label}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  (n={total})")

        if args.also_save_legacy_dir:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_agg, annot=True, fmt=".0f", cmap="Blues",
                        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
            ax.set_title(f"{label} - Aggregated Confusion Matrix (n={folds_used} folds)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            legacy_raw = os.path.join(legacy_dir, f"confusion_matrix_{label}_aggregated.png")
            plt.savefig(legacy_raw, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  {label}: saved {legacy_raw}")

        row_sums = cm_agg.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_agg, row_sums, where=row_sums > 0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, vmin=0, vmax=1)
        ax.set_title(THESIS_TITLES_NORMALIZED[i])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        thesis_path = os.path.join(out_dir, f"confusion_matrix_{label}_aggregated_normalized.png")
        plt.savefig(thesis_path, dpi=300, bbox_inches="tight")
        if args.also_save_legacy_dir:
            legacy_norm = os.path.join(legacy_dir, f"confusion_matrix_{label}_aggregated_normalized.png")
            plt.savefig(legacy_norm, dpi=300, bbox_inches="tight")
            print(f"  {label}: saved {legacy_norm}")
        plt.close()
        print(f"  {label}: saved {thesis_path}")

    # Save accuracy metrics summary
    if all_metrics:
        import csv
        metrics_csv = os.path.join(out_dir, "binary_cnn_accuracy_metrics.csv")
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Folds", "N", "Accuracy", "Precision", "Recall", "F1"])
            writer.writeheader()
            for m in all_metrics:
                writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m.items()})
        print(f"\nAccuracy metrics CSV → {metrics_csv}")

        # Also create a nice bar chart of accuracy metrics
        import pandas as pd
        df = pd.DataFrame(all_metrics)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df))
        width = 0.2
        ax.bar(x - 1.5*width, df["Accuracy"], width, label="Accuracy", color="#3498db")
        ax.bar(x - 0.5*width, df["Precision"], width, label="Precision", color="#2ecc71")
        ax.bar(x + 0.5*width, df["Recall"], width, label="Recall", color="#e74c3c")
        ax.bar(x + 1.5*width, df["F1"], width, label="F1 Score", color="#9b59b6")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Model"], rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Binary Classification Accuracy Metrics (K-Fold Aggregated)")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        metrics_png = os.path.join(out_dir, "binary_cnn_accuracy_metrics.png")
        plt.savefig(metrics_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Accuracy metrics chart → {metrics_png}")

    print(f"\nRow-normalized figures → {out_dir}")
    if args.also_save_legacy_dir:
        print(f"Legacy copies also → {legacy_dir}")


if __name__ == "__main__":
    main()
