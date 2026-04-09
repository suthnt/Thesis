#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

BASE = "/scratch/gpfs/ALAINK/Suthi"
THESIS_DIR = os.path.join(BASE, "Images_For_thesis")

# Regression forest images with legends on upper right
regression_updates = {
    "regression_forest_ordinal.png": {
        "title": "Feature Associations with Crash Counts (Ordinal Logistic Regression)",
        "legend": [
            "Red: More crashes (p < 0.05)",
            "Green: Fewer crashes (p < 0.05)",
            "Gray: No relationship (p ≥ 0.05)",
            "Bars: 95% CI"
        ]
    },
    "regression_forest_binary_crash.png": {
        "title": "Feature Associations with Crash Existence (Binary Logistic Regression)",
        "legend": [
            "Red: Crash occurs (p < 0.05)",
            "Green: No crash (p < 0.05)",
            "Gray: No relationship (p ≥ 0.05)",
            "Bars: 95% CI"
        ]
    }
}

# Confusion matrix titles
confusion_updates = {
    "confusion_matrix_FirstCNN_bin_aggregated_normalized.png": "FirstCNN binary confusion matrix",
    "confusion_matrix_AlexNet_bin_aggregated_normalized.png": "AlexNet binary confusion matrix",
    "confusion_matrix_ResNet50_bin_aggregated_normalized.png": "ResNet50 binary confusion matrix",
    "confusion_matrix_VGG16_1year_aggregated_normalized.png": "VGG16 binary confusion matrix",
    "confusion_matrix_InceptionV3_1year_aggregated_normalized.png": "InceptionV3 binary confusion matrix",
    "confusion_matrix_unet_1year_aggregated_normalized.png": "UNet binary confusion matrix",
}

def update_regression_with_legend(img_path, new_title, legend_lines):
    """
    Load image, extract it, add new title and legend on upper right.
    """
    print(f"Processing: {os.path.basename(img_path)}")
    
    # Load original image
    original_img = Image.open(img_path)
    orig_width, orig_height = original_img.size
    
    # Convert to array
    img_array = np.array(original_img)
    
    # Create figure with the image
    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
    ax.imshow(img_array)
    ax.axis('off')
    
    # Add title at top, spanning full width
    fig.suptitle(new_title, fontsize=14, fontweight='bold', y=0.98)
    
    # Create legend on upper right
    legend_text = "\n".join(legend_lines)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.97, 0.97, legend_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            family='monospace')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(img_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Updated with new title and legend")

def update_confusion_title(img_path, new_title):
    """
    Replace the title of a confusion matrix image.
    """
    print(f"Processing: {os.path.basename(img_path)}")
    
    # Load original image
    original_img = Image.open(img_path)
    img_array = np.array(original_img)
    
    # Create figure with the image
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    ax.imshow(img_array)
    ax.axis('off')
    
    # Add new title
    fig.suptitle(new_title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(img_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Updated title")

def main():
    # Update regression images
    print("=== Updating Regression Forest Images ===")
    for filename, info in regression_updates.items():
        img_path = os.path.join(THESIS_DIR, filename)
        if os.path.exists(img_path):
            update_regression_with_legend(img_path, info["title"], info["legend"])
        else:
            print(f"SKIP: {filename} not found")
    
    # Update confusion matrices
    print("\n=== Updating Confusion Matrix Images ===")
    for filename, new_title in confusion_updates.items():
        img_path = os.path.join(THESIS_DIR, filename)
        if os.path.exists(img_path):
            update_confusion_title(img_path, new_title)
        else:
            print(f"SKIP: {filename} not found")
    
    print("\n✓ All images updated!")

if __name__ == "__main__":
    main()
