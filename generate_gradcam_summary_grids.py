#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import sys
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from PIL import Image

sys.path.insert(0, "/scratch/gpfs/ALAINK/Suthi/yolo_feature_detector/saliency_gradcam_yolo")
from gradcam_unet_multiclass import build_grad_model, make_gradcam_heatmap, preprocess_for_model

BASE = "/scratch/gpfs/ALAINK/Suthi"
OUT_DIR = os.path.join(BASE, "Images_For_thesis")
IMG_SIZE = 224
EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Seed for reproducibility
random.seed(42)
np.random.seed(42)


def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def overlay_heatmap(img_rgb, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on image. Returns uint8 HxWx3."""
    h, w = img_rgb.shape[:2]
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h)))
    heatmap_color = cm.jet(heatmap_resized / 255.0)[:, :, :3]  # drop alpha
    img_norm = img_rgb / 255.0 if img_rgb.max() > 1.5 else img_rgb
    overlay = (1 - alpha) * img_norm + alpha * heatmap_color
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def sample_images(data_dir, class_names, n_per_class=6):
    """Sample n_per_class images from each class directory."""
    samples = []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(EXT)]
        chosen = random.sample(files, min(n_per_class, len(files)))
        for f in chosen:
            samples.append((os.path.join(cls_dir, f), cls))
    return samples


def generate_grid(model_path, data_dir, class_names, class_labels, title,
                  out_path, n_per_class=6):
    """Generate and save a Grad-CAM summary grid."""
    print(f"\nLoading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    grad_model, conv_name = build_grad_model(model)
    print(f"  Grad-CAM layer: {conv_name}")

    samples = sample_images(data_dir, class_names, n_per_class=n_per_class)
    print(f"  Sampled {len(samples)} images")

    n_classes = len(class_names)
    n_cols = n_per_class
    n_rows = n_classes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = [axes]

    idx = 0
    for row, cls in enumerate(class_names):
        cls_samples = [(p, c) for p, c in samples if c == cls]
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(cls_samples):
                ax.axis("off")
                continue

            path, true_label = cls_samples[col]
            img_rgb = load_image_rgb(path)
            x = preprocess_for_model(img_rgb, IMG_SIZE)

            preds = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            conf = float(preds[pred_idx])

            heatmap, _ = make_gradcam_heatmap(x, grad_model, pred_index=pred_idx)
            overlay = overlay_heatmap(img_rgb, heatmap)

            ax.imshow(overlay)
            ax.axis("off")

            true_lbl = class_labels[class_names.index(true_label)]
            pred_lbl = class_labels[pred_idx]

            # Color: green if correct, red if wrong
            color = "#2ecc71" if pred_idx == class_names.index(true_label) else "#e74c3c"
            ax.set_title(f"{true_lbl}\nPred: {pred_lbl} ({conf:.0%})",
                         fontsize=8, color=color, fontweight="bold")
            idx += 1

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Binary U-Net ──
    generate_grid(
        model_path=os.path.join(BASE, "binary_1year_kfold/unet_1year_f0/best_unet_1year.keras"),
        data_dir=os.path.join(BASE, "OrganizedDatasetBalancedBinary_kfold/fold_0/val"),
        class_names=["dangerous", "safe"],
        class_labels=["Dangerous", "Safe"],
        title="Grad-CAM Saliency — U-Net Binary Classifier (Best Fold)",
        out_path=os.path.join(OUT_DIR, "gradcam_summary_unet_binary.png"),
        n_per_class=6,
    )

    # ── Multiclass U-Net ──
    generate_grid(
        model_path=os.path.join(BASE, "balanced_multiclass_1year_kfold/unet_multi_balanced_1year_f0/best_unet_multi_balanced_1year.keras"),
        data_dir=os.path.join(BASE, "OrganizedDatasetMultiClassBalanced_kfold/fold_0/val"),
        class_names=["0_safe", "1_crash", "2_crashes", "3plus_crashes"],
        class_labels=["0 (Safe)", "1 Crash", "2 Crashes", "3+ Crashes"],
        title="Grad-CAM Saliency — U-Net Multiclass Classifier (Best Fold)",
        out_path=os.path.join(OUT_DIR, "gradcam_summary_unet_multiclass.png"),
        n_per_class=6,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
