# This code was written with the assistance of Claude (Anthropic).

import os
import shutil
import random

# === CONFIGURATION ===
SOURCE_DIR = "/scratch/gpfs/ALAINK/Suthi/ChippedImages_5Years"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClass_5Years"
TRAIN_RATIO = 0.8

# Multi-class: use safe, 1_crash, 2_crashes, 3plus_crashes (match OrganizedDatasetMultiClass naming)
CLASS_MAPPING = {
    "safe": "0_safe",
    "1_crash": "1_crash",
    "2_crashes": "2_crashes",
    "3plus_crashes": "3plus_crashes",
}

print("=== Creating Multi-Class Dataset (5 YEARS) ===")
print(f"Source: {SOURCE_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Train ratio: {TRAIN_RATIO}")
print(f"Classes: {list(CLASS_MAPPING.values())}")

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"\n✓ Cleared existing dataset at {OUTPUT_DIR}")

for split in ['train', 'test']:
    for label in CLASS_MAPPING.values():
        os.makedirs(f"{OUTPUT_DIR}/{split}/{label}", exist_ok=True)
print("✓ Created folder structure")


def get_all_images(folder):
    images = []
    if not os.path.isdir(folder):
        return images
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                images.append(os.path.join(root, f))
    return images


def split_and_copy(images, output_label):
    if not images:
        return 0, 0
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    for img in train_images:
        shutil.copy(img, f"{OUTPUT_DIR}/train/{output_label}/{os.path.basename(img)}")
    for img in test_images:
        shutil.copy(img, f"{OUTPUT_DIR}/test/{output_label}/{os.path.basename(img)}")
    return len(train_images), len(test_images)


total_train, total_test = 0, 0
for source_class, output_class in CLASS_MAPPING.items():
    source_path = os.path.join(SOURCE_DIR, source_class)
    images = get_all_images(source_path)
    print(f"Found {len(images)} images in {source_class}")
    if images:
        n_train, n_test = split_and_copy(images, output_class)
        total_train += n_train
        total_test += n_test
        print(f"  -> Train: {n_train}, Test: {n_test}")

print(f"\n=== 5-Year Multi-Class Dataset Created ===")
print(f"Train: {total_train} total")
print(f"Test:  {total_test} total")
print(f"\n✓ Saved to: {OUTPUT_DIR}")
print("\nNext: Run AlexNet_multi_5Years.py, ResNet50_multi_5Years.py, FirstCNN_multi_5Years.py")
