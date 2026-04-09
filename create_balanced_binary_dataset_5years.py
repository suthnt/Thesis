# This code was written with the assistance of Claude (Anthropic).

import os
import shutil
import random

# === CONFIGURATION ===
SAFE_FOLDER = "/scratch/gpfs/ALAINK/Suthi/ChippedImages_5Years/safe"
DANGEROUS_FOLDER = "/scratch/gpfs/ALAINK/Suthi/ChippedImages_5Years/dangerous"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary_5Years"
TRAIN_RATIO = 0.8

print("=== Creating Balanced Binary Dataset (5 YEARS) ===")
print(f"Safe source: {SAFE_FOLDER}")
print(f"Dangerous source: {DANGEROUS_FOLDER}")
print(f"Output: {OUTPUT_DIR}")
print(f"Train ratio: {TRAIN_RATIO}")

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"\n✓ Cleared existing dataset at {OUTPUT_DIR}")

for split in ['train', 'test']:
    for label in ['safe', 'dangerous']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/{label}", exist_ok=True)
print("✓ Created folder structure")

def get_all_images(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                images.append(os.path.join(root, f))
    return images

def split_and_copy(images, label):
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    for img in train_images:
        shutil.copy(img, f"{OUTPUT_DIR}/train/{label}/{os.path.basename(img)}")
    for img in test_images:
        shutil.copy(img, f"{OUTPUT_DIR}/test/{label}/{os.path.basename(img)}")
    return len(train_images), len(test_images)

safe_images = get_all_images(SAFE_FOLDER)
dangerous_images = get_all_images(DANGEROUS_FOLDER)

print(f"\nFound {len(safe_images)} safe images")
print(f"Found {len(dangerous_images)} dangerous images")

# Balance by undersampling dangerous to match safe
random.shuffle(dangerous_images)
dangerous_images = dangerous_images[:len(safe_images)]
print(f"After balancing: {len(safe_images)} safe, {len(dangerous_images)} dangerous")

safe_train, safe_test = split_and_copy(safe_images, 'safe')
danger_train, danger_test = split_and_copy(dangerous_images, 'dangerous')

print(f"\n=== 5-Year Balanced Dataset Created ===")
print(f"Train: {safe_train} safe + {danger_train} dangerous = {safe_train + danger_train} total")
print(f"Test:  {safe_test} safe + {danger_test} dangerous = {safe_test + danger_test} total")
print(f"\n✓ Saved to: {OUTPUT_DIR}")
