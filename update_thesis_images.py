#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import shutil
from PIL import Image, ImageDraw, ImageFont

BASE = "/scratch/gpfs/ALAINK/Suthi"
SOURCE_DIR = os.path.join(BASE, "yolo_feature_detector/plots_regression_prevalence_suite/yolo_pr_exp2")
THESIS_DIR = os.path.join(BASE, "Images_For_thesis")

files_to_process = [
    {
        "source": "regression_forest.png",
        "target": "regression_forest_ordinal.png",
        "title": "Feature Associations with Crash Counts (Ordinal Logistic Regression)",
        "legend": [
            "Red: statistically significant correlation for more crashes (p < 0.05)",
            "Green: statistically significant correlation for fewer crashes (p < 0.05)",
            "Gray: no significant relationship (p ≥ 0.05)",
            "Bars represent 95% confidence intervals"
        ]
    },
    {
        "source": "regression_forest_binary.png",
        "target": "regression_forest_binary_crash.png",
        "title": "Feature Associations with Crash Existence (Binary Logistic Regression)",
        "legend": [
            "Red: statistically significant correlation with crash occurrence (p < 0.05)",
            "Green: statistically significant correlation with crash absence (p < 0.05)",
            "Gray: no significant relationship (p ≥ 0.05)",
            "Bars represent 95% confidence intervals"
        ]
    }
]

def add_title_and_legend(img_path, new_title, legend_text):
    """
    Add a new title and legend to the image.
    """
    # Load the original image
    img = Image.open(img_path)
    width, height = img.size
    
    # Calculate space needed for title and legend
    title_height = 60  # Space for title
    legend_height = len(legend_text) * 25 + 30  # Space for legend lines
    top_padding = 10
    bottom_padding = 20
    left_padding = 20
    
    # Create new image with extra space
    new_height = height + title_height + legend_height + top_padding + bottom_padding
    new_img = Image.new('RGB', (width, new_height), 'white')
    
    # Paste original image below title/legend space
    new_img.paste(img, (0, title_height))
    
    # Draw on the new image
    draw = ImageDraw.Draw(new_img)
    
    # Try to load a font, fall back to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        legend_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()
    
    # Draw title
    title_y = 10
    draw.text((left_padding, title_y), new_title, fill='black', font=title_font)
    
    # Draw legend
    legend_start_y = height + title_height + 5
    draw.text((left_padding, legend_start_y), "Legend:", fill='black', font=legend_font)
    
    colors = ['red', 'green', 'gray', 'black']
    for i, line in enumerate(legend_text):
        y = legend_start_y + 25 + i * 25
        # Draw color box
        box_size = 12
        color = colors[i]
        draw.rectangle([left_padding, y, left_padding + box_size, y + box_size], fill=color)
        # Draw text
        draw.text((left_padding + box_size + 8, y), line, fill='black', font=legend_font)
    
    return new_img


def main():
    os.makedirs(THESIS_DIR, exist_ok=True)
    
    for item in files_to_process:
        source_path = os.path.join(SOURCE_DIR, item["source"])
        target_path = os.path.join(THESIS_DIR, item["target"])
        
        if not os.path.exists(source_path):
            print(f"ERROR: Source file not found: {source_path}")
            continue
        
        print(f"Processing: {item['source']} -> {item['target']}")
        
        # Add title and legend
        modified_img = add_title_and_legend(source_path, item["title"], item["legend"])
        
        # Save to thesis folder
        modified_img.save(target_path, quality=95)
        print(f"  ✓ Saved to {target_path}")


if __name__ == "__main__":
    main()
