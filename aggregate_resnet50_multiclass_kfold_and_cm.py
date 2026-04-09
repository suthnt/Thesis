#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

BASE = "/scratch/gpfs/ALAINK/Suthi/balanced_multiclass_1year_kfold"
MODEL = "ResNet50_multi_balanced_1year"
K = 5

# Aggregate validation accuracies
accs = []
all_true = []
all_pred = []
for fold in range(K):
    out_dir = os.path.join(BASE, f"{MODEL}_f{fold}")
    result_path = os.path.join(out_dir, "kfold_result.json")
    preds_path = os.path.join(out_dir, "val_predictions.npz")
    if os.path.exists(result_path):
        with open(result_path) as f:
            accs.append(json.load(f)["val_accuracy"])
    if os.path.exists(preds_path):
        npz = np.load(preds_path)
        all_true.append(npz["y_true"])
        all_pred.append(npz["y_pred"])

if accs:
    print(f"ResNet50 mean val acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} (n={len(accs)} folds)")
else:
    print("No accuracy results found.")

if all_true and all_pred:
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv("/scratch/gpfs/ALAINK/Suthi/Images_For_thesis/resnet50_multiclass_cm_normalized.csv")
    print("Confusion matrix saved to Images_For_thesis/resnet50_multiclass_cm_normalized.csv")
else:
    print("No prediction files found for confusion matrix.")
