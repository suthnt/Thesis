#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import sys
import glob
import numpy as np
import cv2
from pathlib import Path

# Add Mask2Former to path (clone repo to project dir first)
MASK2FORMER_ROOT = "/scratch/gpfs/ALAINK/Suthi/Mask2Former"
if not os.path.isdir(MASK2FORMER_ROOT):
    raise FileNotFoundError(
        f"Mask2Former not found at {MASK2FORMER_ROOT}. "
        "Clone with: git clone https://github.com/facebookresearch/Mask2Former.git"
    )
sys.path.insert(0, MASK2FORMER_ROOT)

# === CONFIGURATION ===
# Cityscapes: street/road pretrained - 19 classes (road, car, pedestrian, bicycle, etc.)
# More task-relevant for intersection safety than COCO.
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary"  # 1-year (normal)
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/mask2former_features_1year_cityscapes"
CONFIG_FILE = os.path.join(MASK2FORMER_ROOT, "configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml")
CHECKPOINT_DIR = os.path.join(MASK2FORMER_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MODEL_WEIGHTS = os.path.join(CHECKPOINT_DIR, "maskformer2_R50_Cityscapes_semantic.pkl")
# Official URLs from Mask2Former MODEL_ZOO.md (Cityscapes semantic R50)
MODEL_WEIGHTS_URLS = [
    "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R50_bs16_90k/model_final_cc1b1f.pkl",
]
MIN_CHECKPOINT_SIZE = 80_000_000  # ~80MB - R50 model; reject truncated downloads

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"=== Mask2Former Feature Extraction (1-Year Dataset, Cityscapes) ===")
print(f"Dataset: {DATASET_DIR}")
print(f"Output: {OUTPUT_DIR}")


def ensure_model_weights():
    """Download model weights if not present or corrupted."""
    if os.path.isfile(MODEL_WEIGHTS) and os.path.getsize(MODEL_WEIGHTS) >= MIN_CHECKPOINT_SIZE:
        return MODEL_WEIGHTS
    if os.path.isfile(MODEL_WEIGHTS) and os.path.getsize(MODEL_WEIGHTS) < MIN_CHECKPOINT_SIZE:
        print(f"Removing corrupted/empty checkpoint ({os.path.getsize(MODEL_WEIGHTS)} bytes)")
        os.remove(MODEL_WEIGHTS)
    for i, url in enumerate(MODEL_WEIGHTS_URLS):
        print(f"Downloading Mask2Former R50 Cityscapes weights (attempt {i+1}/{len(MODEL_WEIGHTS_URLS)})...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, MODEL_WEIGHTS)
            if os.path.getsize(MODEL_WEIGHTS) >= MIN_CHECKPOINT_SIZE:
                return MODEL_WEIGHTS
            os.remove(MODEL_WEIGHTS)
        except Exception as e:
            print(f"  Failed: {e}")
    raise RuntimeError(
        "Could not download model. Manually download from:\n"
        "  https://github.com/facebookresearch/Mask2Former (see MODEL_ZOO.md)\n"
        f"  Save to: {MODEL_WEIGHTS}"
    )


def setup_cfg():
    """Setup Mask2Former config from repo."""
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(CONFIG_FILE)
    weights_path = ensure_model_weights()
    cfg.merge_from_list(["MODEL.WEIGHTS", weights_path])
    cfg.freeze()
    return cfg


def extract_features_from_split(split: str, predictor):
    """Extract features for train or test split."""
    split_dir = os.path.join(DATASET_DIR, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    all_features = []
    all_labels = []
    all_filenames = []

    for label_name in ["safe", "dangerous"]:
        label_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        label_idx = 0 if label_name == "safe" else 1

        images = glob.glob(os.path.join(label_dir, "*.png")) + glob.glob(os.path.join(label_dir, "*.jpg")) + \
                 glob.glob(os.path.join(label_dir, "*.jpeg")) + glob.glob(os.path.join(label_dir, "*.tif"))
        images = sorted(images)

        print(f"  {split}/{label_name}: {len(images)} images")
        for path in images:
            img = cv2.imread(path)
            if img is None:
                print(f"    Skip (cannot read): {path}")
                continue
            # BGR format expected by detectron2
            predictions = predictor(img)
            if "sem_seg" in predictions:
                sem_seg = predictions["sem_seg"]  # [num_classes, H, W]
                # Global average pool -> [num_classes]
                feat = sem_seg.cpu().numpy().mean(axis=(1, 2))
            else:
                # Fallback: use panoptic or instance if sem_seg missing
                print(f"    Warning: no sem_seg for {path}, skipping")
                continue

            all_features.append(feat)
            all_labels.append(label_idx)
            all_filenames.append(os.path.basename(path))

    if len(all_features) == 0:
        raise RuntimeError(f"No features extracted for {split}")

    return np.array(all_features, dtype=np.float32), np.array(all_labels), all_filenames


def main():
    cfg = setup_cfg()
    from detectron2.engine import DefaultPredictor

    predictor = DefaultPredictor(cfg)
    print("Model loaded.")

    # Process train and test
    for split in ["train", "test"]:
        print(f"\nExtracting {split}...")
        features, labels, filenames = extract_features_from_split(split, predictor)
        out_path = os.path.join(OUTPUT_DIR, f"{split}_features.npz")
        np.savez_compressed(out_path, features=features, labels=labels, filenames=filenames)
        print(f"  Saved: {out_path} (shape {features.shape})")

    # Summary
    train_data = np.load(os.path.join(OUTPUT_DIR, "train_features.npz"))
    test_data = np.load(os.path.join(OUTPUT_DIR, "test_features.npz"))
    print(f"\n=== Done ===")
    print(f"Train: {train_data['features'].shape[0]} samples, {train_data['features'].shape[1]} dims")
    print(f"Test:  {test_data['features'].shape[0]} samples")
    print(f"Use: features, labels = data['features'], data['labels']")
    print(f"Example: from sklearn.linear_model import LogisticRegression")
    print(f"         clf = LogisticRegression().fit(train_data['features'], train_data['labels'])")

if __name__ == "__main__":
    main()
