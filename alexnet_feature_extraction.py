# This code was written with the assistance of Claude (Anthropic).

import os
import glob
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model

# === CONFIG ===
MODEL_PATH = "/scratch/gpfs/ALAINK/Suthi/AlexNet_bin.keras"
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/alexnet_features_1year"
IMG_SIZE = 227
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== AlexNet Feature Extraction (last layer before softmax) ===")
print(f"Model: {MODEL_PATH}")
print(f"Dataset: {DATASET_DIR}")
print(f"Output: {OUTPUT_DIR}")

# Load full model
model = load_model(MODEL_PATH)

# AlexNet: ... Dense(4096), Dropout, Dense(4096), Dropout, Dense(2)
feature_layer = model.layers[-3]
if feature_layer.output.shape[-1] != 4096:
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if "dense" in layer.name.lower() and layer.output.shape[-1] == 4096:
            feature_layer = layer
            break
print(f"Extracting from layer: {feature_layer.name} (output dim={feature_layer.output.shape[-1]})")

# Build feature extractor
_ = model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)), verbose=0)
feature_model = Model(inputs=model.inputs[0], outputs=feature_layer.output)


def extract_split(split):
    """Load images manually (avoids flow_from_directory multiprocessing issues)."""
    split_dir = os.path.join(DATASET_DIR, split)
    all_features = []
    all_labels = []
    all_filenames = []

    for label_name in ["safe", "dangerous"]:
        label_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        label_idx = 0 if label_name == "safe" else 1

        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif")
        images = []
        for ext in exts:
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
                img = img_to_array(img)
                img = img.astype(np.float32) / 255.0
                batch_imgs.append(img)
                all_filenames.append(os.path.basename(path))
                all_labels.append(label_idx)

            if batch_imgs:
                X = np.array(batch_imgs)
                feats = feature_model.predict(X, verbose=0)
                all_features.append(feats)

    if not all_features:
        raise RuntimeError(f"No features extracted for {split}")

    return np.vstack(all_features), np.array(all_labels), all_filenames


# Extract
for split in ["train", "test"]:
    print(f"\nExtracting {split}...")
    features, labels, filenames = extract_split(split)
    out_path = os.path.join(OUTPUT_DIR, f"{split}_features.npz")
    np.savez_compressed(out_path, features=features, labels=labels, filenames=filenames)
    print(f"  Saved: {out_path} (shape {features.shape})")

# Summary
train_data = np.load(os.path.join(OUTPUT_DIR, "train_features.npz"))
test_data = np.load(os.path.join(OUTPUT_DIR, "test_features.npz"))
print(f"\n=== Done ===")
print(f"Train: {train_data['features'].shape[0]} samples, {train_data['features'].shape[1]} dims")
print(f"Test:  {test_data['features'].shape[0]} samples")
print("Use: data = np.load(...); X, y = data['features'], data['labels']")
