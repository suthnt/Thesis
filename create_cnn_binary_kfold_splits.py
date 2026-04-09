#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import random
import shutil

BASE = "/scratch/gpfs/ALAINK/Suthi"
SOURCE = os.path.join(BASE, "OrganizedDatasetBalancedBinary")
OUT_BASE = os.path.join(BASE, "OrganizedDatasetBalancedBinary_kfold")
K = 5
SEED = 42

CLASSES = ["safe", "dangerous"]
EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def main():
    # Collect all (split, class, filename) from train + test
    all_items = []
    for split in ["train", "test"]:
        for cls in CLASSES:
            folder = os.path.join(SOURCE, split, cls)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if f.lower().endswith(EXT):
                    all_items.append((split, cls, f))

    # Stratified split by class (manual, no sklearn)
    random.seed(SEED)
    by_class = {}
    for s, c, f in all_items:
        by_class.setdefault(c, []).append((s, c, f))
    for c in by_class:
        random.shuffle(by_class[c])

    folds = [[] for _ in range(K)]
    for cls, items in by_class.items():
        for i, item in enumerate(items):
            folds[i % K].append(item)

    n = len(all_items)
    print(f"Total: {n} images, {K} folds (stratified by class)")

    for fold_idx in range(K):
        val_set = set((s, c, f) for s, c, f in folds[fold_idx])
        train_items = [(s, c, f) for i, fold in enumerate(folds) if i != fold_idx for s, c, f in fold]

        fold_dir = os.path.join(OUT_BASE, f"fold_{fold_idx}")
        for split in ["train", "val"]:
            for cls in CLASSES:
                os.makedirs(os.path.join(fold_dir, split, cls), exist_ok=True)

        for s, c, f in all_items:
            dst_split = "val" if (s, c, f) in val_set else "train"
            src = os.path.join(SOURCE, s, c, f)
            dst = os.path.join(fold_dir, dst_split, c, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        n_train = len(train_items)
        n_val = len(val_set)
        print(f"  fold_{fold_idx}: train={n_train}, val={n_val}")

    print(f"\nCreated {OUT_BASE}/fold_0 ... fold_4")


if __name__ == "__main__":
    main()
