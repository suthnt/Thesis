#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import json
import os

BASE = "/scratch/gpfs/ALAINK/Suthi/binary_1year_kfold"
MODELS = [
    ("FirstCNN_bin_f", "FirstCNN"),
    ("AlexNet_bin_f", "AlexNet"),
    ("ResNet50_bin_f", "ResNet50"),
    ("VGG16_1year_f", "VGG16"),
    ("InceptionV3_1year_f", "InceptionV3"),
    ("unet_1year_f", "U-Net"),
]
K = 5


def main():
    print("\n" + "=" * 60)
    print("Binary CNN K-Fold Cross-Validation Results (k=5)")
    print("=" * 60)

    for dir_prefix, label in MODELS:
        accs = []
        for fold in range(K):
            out_dir = os.path.join(BASE, f"{dir_prefix}{fold}")
            path = os.path.join(out_dir, "kfold_result.json")
            if not os.path.exists(path):
                print(f"  {label} fold {fold}: not found")
                continue
            with open(path) as f:
                data = json.load(f)
            accs.append(data["val_accuracy"])

        if accs:
            avg = sum(accs) / len(accs)
            std = (sum((a - avg) ** 2 for a in accs) / len(accs)) ** 0.5
            print(f"\n{label}: val_acc = {avg:.4f} ± {std:.4f} (n={len(accs)} folds)")

    print("")


if __name__ == "__main__":
    main()
