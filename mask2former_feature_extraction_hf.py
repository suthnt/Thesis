#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import glob
import numpy as np
from PIL import Image

# === CONFIGURATION ===
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/mask2former_features_1year"
# Swin-small is fast; swin-large for better features
MODEL_NAME = "facebook/mask2former-swin-small-ade-semantic"  # 150 semantic classes

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"=== Mask2Former Feature Extraction (Hugging Face) ===")
print(f"Dataset: {DATASET_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Model: {MODEL_NAME}")

def extract_features_from_split(split: str, processor, model):
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

        images = sorted(glob.glob(os.path.join(label_dir, "*.png")) + 
                        glob.glob(os.path.join(label_dir, "*.jpg")) +
                        glob.glob(os.path.join(label_dir, "*.jpeg")))
        print(f"  {split}/{label_name}: {len(images)} images")

        for path in images:
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                print(f"    Skip (cannot read): {path}")
                continue

            import torch
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Semantic: use class_queries_logits [batch, num_queries, num_classes], pool over queries
            logits = outputs.class_queries_logits  # [1, N, num_classes]
            feat = logits.mean(dim=1).cpu().numpy().squeeze().astype(np.float32)

            all_features.append(feat)
            all_labels.append(label_idx)
            all_filenames.append(os.path.basename(path))

    if len(all_features) == 0:
        raise RuntimeError(f"No features extracted for {split}")

    return np.array(all_features), np.array(all_labels), all_filenames


def main():
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from cache (device={device})...")
    # Compute nodes have no network - use cache from login-node pre-download:
    #   conda activate m2f_hf && python download_m2f_hf_model.py
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME, local_files_only=True).to(device)
    except OSError as e:
        if "Can't load" in str(e) or "Connection" in str(e):
            raise SystemExit(
                "Model not in cache. On a LOGIN NODE run:\n"
                "  conda activate m2f_hf && python download_m2f_hf_model.py\n"
                "Then re-submit this job."
            ) from e
        raise
    model.eval()
    print("Model loaded.")

    for split in ["train", "test"]:
        print(f"\nExtracting {split}...")
        features, labels, filenames = extract_features_from_split(split, processor, model)
        out_path = os.path.join(OUTPUT_DIR, f"{split}_features.npz")
        np.savez_compressed(out_path, features=features, labels=labels, filenames=filenames)
        print(f"  Saved: {out_path} (shape {features.shape})")

    train_data = np.load(os.path.join(OUTPUT_DIR, "train_features.npz"))
    test_data = np.load(os.path.join(OUTPUT_DIR, "test_features.npz"))
    print(f"\n=== Done ===")
    print(f"Train: {train_data['features'].shape[0]} samples, {train_data['features'].shape[1]} dims")
    print(f"Test:  {test_data['features'].shape[0]} samples")


if __name__ == "__main__":
    main()
