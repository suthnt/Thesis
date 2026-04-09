# This code was written with the assistance of Claude (Anthropic).

import os
import glob
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model

# === CONFIG ===
MODEL_PATH = "/scratch/gpfs/ALAINK/Suthi/best_unet_1year.keras"
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/unet_features_1year"
IMG_SIZE = 224
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== U-Net Feature Extraction ===")
print(f"Model: {MODEL_PATH}")
print(f"Dataset: {DATASET_DIR}")
print(f"Output: {OUTPUT_DIR}")

model = load_model(MODEL_PATH)
# U-Net: ... GAP -> fc1 (256) -> Dropout -> output (2)
# Extract from fc1 (256 dims)
feature_layer = model.get_layer("fc1")
_ = model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)), verbose=0)
feature_model = Model(inputs=model.inputs[0], outputs=feature_layer.output)
print(f"Extracting from: {feature_layer.name} (dim={feature_layer.output.shape[-1]})")


def extract_split(split):
    split_dir = os.path.join(DATASET_DIR, split)
    all_features, all_labels, all_filenames = [], [], []
    for label_name in ["safe", "dangerous"]:
        label_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        label_idx = 0 if label_name == "safe" else 1
        images = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif"):
            images.extend(glob.glob(os.path.join(label_dir, ext)))
        images = sorted(images)
        print(f"  {split}/{label_name}: {len(images)} images")
        for i in range(0, len(images), BATCH_SIZE):
            batch_paths = images[i : i + BATCH_SIZE]
            batch_imgs = []
            for path in batch_paths:
                try:
                    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
                except Exception:
                    continue
                img = img_to_array(img).astype(np.float32) / 255.0
                batch_imgs.append(img)
                all_filenames.append(os.path.basename(path))
                all_labels.append(label_idx)
            if batch_imgs:
                X = np.array(batch_imgs)
                feats = feature_model.predict(X, verbose=0)
                all_features.append(feats)
    if not all_features:
        raise RuntimeError(f"No features for {split}")
    return np.vstack(all_features), np.array(all_labels), all_filenames


for split in ["train", "test"]:
    print(f"\nExtracting {split}...")
    features, labels, filenames = extract_split(split)
    out_path = os.path.join(OUTPUT_DIR, f"{split}_features.npz")
    np.savez_compressed(out_path, features=features, labels=labels, filenames=filenames)
    print(f"  Saved: {out_path} ({features.shape})")

print("\n=== Done ===")
print(f"Add to meta_classifier FEATURE_DIRS: \"{OUTPUT_DIR}\"")
