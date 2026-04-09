#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
from PIL import Image, ImageDraw, ImageFont

BASE = "/scratch/gpfs/ALAINK/Suthi"
THESIS_DIR = os.path.join(BASE, "Images_For_thesis")

# Map old filenames to new titles
title_updates = {
    "confusion_matrix_FirstCNN_bin_aggregated_normalized.png": "FirstCNN binary confusion matrix",
    "confusion_matrix_AlexNet_bin_aggregated_normalized.png": "AlexNet binary confusion matrix",
    "confusion_matrix_ResNet50_bin_aggregated_normalized.png": "ResNet50 binary confusion matrix",
    "confusion_matrix_VGG16_1year_aggregated_normalized.png": "VGG16 binary confusion matrix",
    "confusion_matrix_InceptionV3_1year_aggregated_normalized.png": "InceptionV3 binary confusion matrix",
    "confusion_matrix_unet_1year_aggregated_normalized.png": "UNet binary confusion matrix",
}

def add_title_to_image(img_path, new_title):
    """
    Add a new title to the top of the image using PIL.
    """
    img = Image.open(img_path)
    width, height = img.size
    
    # Create space for title at top
    title_height = 50
    new_height = height + title_height
    new_img = Image.new('RGB', (width, new_height), 'white')
    
    # Paste original image below title
    new_img.paste(img, (0, title_height))
    
    # Draw title
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Center the title
    bbox = draw.textbbox((0, 0), new_title, font=font)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    
    draw.text((x, 15), new_title, fill='black', font=font)
    
    return new_img

def main():
    for filename, new_title in title_updates.items():
        img_path = os.path.join(THESIS_DIR, filename)
        
        if not os.path.exists(img_path):
            print(f"SKIP: {filename} not found")
            continue
        
        print(f"Updating: {filename}")
        
        # Create backup
        backup_path = img_path.replace(".png", "_original.png")
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy(img_path, backup_path)
            print(f"  Backup: {backup_path}")
        
        # Update with new title
        modified_img = add_title_to_image(img_path, new_title)
        modified_img.save(img_path, quality=95)
        print(f"  ✓ Updated with new title")

if __name__ == "__main__":
    main()
