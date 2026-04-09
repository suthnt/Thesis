# This code was written with the assistance of Claude (Anthropic).

import os
import shutil
import random
from PIL import Image

# === CONFIGURATION ===
SOURCE_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClass"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClassBalanced"
TRAIN_RATIO = 0.8
CLASSES = ["0_safe", "1_crash", "2_crashes", "3plus_crashes"]

print("=== Creating Balanced Multi-Class Dataset (1-year) ===")
print(f"Source: {SOURCE_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("Strategy: Equal samples per class + 90/180/270° rotations (4x data)")
print(f"Classes: {CLASSES}")

# Count images per class
class_counts = {}
for split in ["train", "test"]:
    class_counts[split] = {}
    for cls in CLASSES:
        path = os.path.join(SOURCE_DIR, split, cls)
        if os.path.isdir(path):
            imgs = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            class_counts[split][cls] = imgs
        else:
            class_counts[split][cls] = []

min_train = min(len(class_counts["train"][c]) for c in CLASSES)
min_test = min(len(class_counts["test"][c]) for c in CLASSES)
print(f"\nMin train per class: {min_train}")
print(f"Min test per class: {min_test}")

# Sample N per class (N = min)
N_TRAIN = min_train
N_TEST = min_test

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"\n✓ Cleared existing dataset at {OUTPUT_DIR}")

for split in ["train", "test"]:
    for cls in CLASSES:
        os.makedirs(f"{OUTPUT_DIR}/{split}/{cls}", exist_ok=True)
print("✓ Created folder structure")


def rotate_and_save(src_path, out_dir, base_name, ext):
    """Save original + 90, 180, 270 degree rotations."""
    try:
        img = Image.open(src_path).convert("RGB")
    except Exception as e:
        print(f"  Skip {src_path}: {e}")
        return 0
    count = 0
    for angle, suffix in [(0, ""), (90, "_r90"), (180, "_r180"), (270, "_r270")]:
        if angle == 0:
            out_img = img
        else:
            out_img = img.rotate(-angle, expand=True)
        out_name = f"{base_name}{suffix}{ext}"
        out_path = os.path.join(out_dir, out_name)
        out_img.save(out_path, quality=95)
        count += 1
    return count


def process_split(split, n_per_class, add_rotations=False):
    """add_rotations: if True, create 90/180/270 copies (4x data). Use for train only."""
    total = 0
    for cls in CLASSES:
        imgs = class_counts[split][cls]
        if len(imgs) < n_per_class:
            print(f"  WARNING: {cls} has only {len(imgs)} images, need {n_per_class}")
            selected = imgs
        else:
            selected = random.sample(imgs, n_per_class)
        out_dir = f"{OUTPUT_DIR}/{split}/{cls}"
        for f in selected:
            src = os.path.join(SOURCE_DIR, split, cls, f)
            base, ext = os.path.splitext(f)
            if add_rotations:
                n = rotate_and_save(src, out_dir, base, ext)
            else:
                # Test set: just copy original (no augmentation)
                try:
                    shutil.copy(src, os.path.join(out_dir, f))
                    total += 1
                except Exception as e:
                    print(f"  Skip {src}: {e}")
                continue
            total += n
        mult = 4 if add_rotations else 1
        print(f"  {cls}: {len(selected)} originals -> {len(selected)*mult} images")
    return total


random.seed(42)
print("\nProcessing train (with 90/180/270 rotations = 4x data)...")
train_total = process_split("train", N_TRAIN, add_rotations=True)
print(f"Train total: {train_total}")

print("\nProcessing test (originals only, no augmentation)...")
test_total = process_split("test", N_TEST, add_rotations=False)
print(f"Test total: {test_total}")

print(f"\n=== Balanced Multi-Class Dataset Created ===")
print(f"Train: {train_total} images ({N_TRAIN} per class × 4 rotations = 4x data)")
print(f"Test:  {test_total} images ({N_TEST} per class, originals only)")
print(f"\n✓ Saved to: {OUTPUT_DIR}")
print("\nNext: Run balanced multi-class training scripts (unet_multi_balanced_1year.py, etc.)")
