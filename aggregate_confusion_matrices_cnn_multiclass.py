#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

BASE = "/scratch/gpfs/ALAINK/Suthi"
K = 5
CLASSES = ["0_safe", "1_crash", "2_crashes", "3plus_crashes"]

MODELS = [
    ("FirstCNN_multi_balanced_1year", "FirstCNN", 99, "best_FirstCNN_multi_balanced_1year.keras"),
    ("AlexNet_multi_balanced_1year", "AlexNet", 227, "best_AlexNet_multi_balanced_1year.keras"),
    ("InceptionV3_multi_balanced_1year", "InceptionV3", 299, "best_InceptionV3_multi_balanced_1year.keras"),
    ("VGG16_multi_balanced_1year", "VGG16", 224, "best_VGG16_multi_balanced_1year.keras"),
    ("unet_multi_balanced_1year", "U-Net", 224, "best_unet_multi_balanced_1year.keras"),
]

OUT_DIR = os.path.join(BASE, "aggregated_confusion_matrices", "cnn_multiclass")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    for dir_prefix, label, img_size, ckpt_name in MODELS:
        cm_agg = np.zeros((len(CLASSES), len(CLASSES)))
        folds_used = 0


        for fold in range(K):
            model_path = os.path.join(BASE, "balanced_multiclass_1year_kfold", f"{dir_prefix}_f{fold}", ckpt_name)
            val_dir = os.path.join(BASE, "OrganizedDatasetMultiClassBalanced_kfold", f"fold_{fold}", "val")

            missing = []
            if not os.path.exists(model_path):
                missing.append(f"model checkpoint: {model_path}")
            if not os.path.exists(val_dir):
                missing.append(f"validation data: {val_dir}")
            if missing:
                print(f"  {label} fold {fold}: SKIP - missing: {', '.join(missing)}")
                continue

            print(f"  {label} fold {fold}: loading model and data...")
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
            cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
            cm_agg += cm
            folds_used += 1

        if folds_used == 0:
            print(f"  {label}: no folds found, skip")
            continue


        # Save raw aggregated CM as .npy
        out_path_npy = os.path.join(OUT_DIR, f"confusion_matrix_{label}_aggregated.npy")
        np.save(out_path_npy, cm_agg)
        print(f"  {label}: saved {out_path_npy}")

        # Save raw aggregated CM as .csv
        out_path_csv = os.path.join(OUT_DIR, f"confusion_matrix_{label}_aggregated.csv")
        np.savetxt(out_path_csv, cm_agg, delimiter=",", fmt="%d")
        print(f"  {label}: saved {out_path_csv}")

        # Raw aggregated CM plot
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm_agg, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title(f"{label} - Aggregated Confusion Matrix (n={folds_used} folds)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"confusion_matrix_{label}_aggregated.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  {label}: saved {out_path}")

        # Row-normalized (recall per class)
        row_sums = cm_agg.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_agg, row_sums, where=row_sums > 0)
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, vmin=0, vmax=1)
        ax.set_title(f"{label} - Aggregated CM (Normalized by Row = Recall)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        out_path_norm = os.path.join(OUT_DIR, f"confusion_matrix_{label}_aggregated_normalized.png")
        plt.savefig(out_path_norm, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  {label}: saved {out_path_norm}")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
