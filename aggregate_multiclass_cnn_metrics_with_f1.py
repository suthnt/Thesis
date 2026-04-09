#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_KFOLD = "/scratch/gpfs/ALAINK/Suthi/balanced_multiclass_1year_kfold"
BASE_DATA = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClassBalanced_kfold"
CLASSES = ["0_safe", "1_crash", "2_crashes", "3plus_crashes"]

MODELS = [
    ("FirstCNN_multi_balanced_1year", "FirstCNN", 99, "best_FirstCNN_multi_balanced_1year.keras"),
    ("AlexNet_multi_balanced_1year", "AlexNet", 227, "best_AlexNet_multi_balanced_1year.keras"),
    ("InceptionV3_multi_balanced_1year", "InceptionV3", 299, "best_InceptionV3_multi_balanced_1year.keras"),
    ("VGG16_multi_balanced_1year", "VGG16", 224, "best_VGG16_multi_balanced_1year.keras"),
    ("unet_multi_balanced_1year", "U-Net", 224, "best_unet_multi_balanced_1year.keras"),
    ("ResNet50_multi_balanced_1year", "ResNet50", 224, "best_ResNet50_multi_balanced_1year.keras"),
]
K = 5

out = []
for dir_prefix, label, img_size, ckpt_name in MODELS:
    accs, f1_macro, f1_micro, f1_weighted = [], [], [], []
    for fold in range(K):
        fold_dir = os.path.join(BASE_KFOLD, f"{dir_prefix}_f{fold}")
        model_path = os.path.join(fold_dir, ckpt_name)
        val_dir = os.path.join(BASE_DATA, f"fold_{fold}", "val")

        # Try loading cached predictions first
        preds_path = os.path.join(fold_dir, "val_predictions.npz")
        if os.path.exists(preds_path):
            print(f"  {label} fold {fold}: loading cached predictions")
            npz = np.load(preds_path)
            y_true = npz["y_true"]
            y_pred = npz["y_pred"]
        elif os.path.exists(model_path) and os.path.exists(val_dir):
            print(f"  {label} fold {fold}: running inference...")
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
            # Cache predictions for future use
            np.savez(preds_path, y_true=y_true, y_pred=y_pred)
            print(f"  {label} fold {fold}: saved predictions to {preds_path}")
        else:
            print(f"  {label} fold {fold}: SKIP (missing model or val data)")
            continue

        accs.append(accuracy_score(y_true, y_pred))
        f1_macro.append(f1_score(y_true, y_pred, average="macro"))
        f1_micro.append(f1_score(y_true, y_pred, average="micro"))
        f1_weighted.append(f1_score(y_true, y_pred, average="weighted"))

    if accs:
        out.append([
            label,
            np.mean(accs), np.std(accs),
            np.mean(f1_macro), np.std(f1_macro),
            np.mean(f1_micro), np.std(f1_micro),
            np.mean(f1_weighted), np.std(f1_weighted),
            len(accs)
        ])
    else:
        out.append([label] + ["NA"]*9 + [0])

with open("/scratch/gpfs/ALAINK/Suthi/Images_For_thesis/multiclass_cnn_accuracy_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "model",
        "mean_val_accuracy", "std_val_accuracy",
        "mean_f1_macro", "std_f1_macro",
        "mean_f1_micro", "std_f1_micro",
        "mean_f1_weighted", "std_f1_weighted",
        "n_folds"
    ])
    for row in out:
        w.writerow(row)
print("Done. Wrote accuracy and F1 scores to multiclass_cnn_accuracy_metrics.csv")
