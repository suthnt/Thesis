#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE = "/scratch/gpfs/ALAINK/Suthi"
SRC_DIR_RECALL = os.path.join(BASE, "aggregated_confusion_matrices", "yolo")
DST_DIR = os.path.join(BASE, "Images_For_thesis")
os.makedirs(DST_DIR, exist_ok=True)

# True Recall CMs: (source filename, display name, output filename)
TRUE_RECALL_FILES = [
    ("confusion_matrix_yolov8n_aggregated_recall.png",      "YOLOv8n-OBB True Recall Confusion Matrix",           "yolo_n_true_recall.png"),
    ("confusion_matrix_yolov8s_aggregated_recall.png",      "YOLOv8s-OBB True Recall Confusion Matrix",           "yolo_s_true_recall.png"),
    ("confusion_matrix_yolov8m_aggregated_recall.png",      "YOLOv8m-OBB True Recall Confusion Matrix",           "yolo_m_true_recall.png"),
    ("confusion_matrix_yolov8m_exp2_aggregated_recall.png", "YOLOv8m-OBB (Exp 2) True Recall Confusion Matrix",   "yolo_m_exp2_true_recall.png"),
    ("confusion_matrix_yolov8l_aggregated_recall.png",      "YOLOv8l-OBB True Recall Confusion Matrix",           "yolo_l_true_recall.png"),
]


def retitle_image(src_path: str, dst_path: str, new_title: str):
    """Load image, blank old title+subtitle, render new title via matplotlib, save."""
    img = mpimg.imread(src_path)
    h, w = img.shape[:2]

    # Blank out old title + subtitle region (2 lines, ~rows 0 to 115 for ~2970px image)
    title_bottom = int(h * 0.04)  # ~119 px for 2971 — covers both title lines
    img[:title_bottom, :] = 1.0  # white

    fig, ax = plt.subplots(figsize=(w / 300, h / 300), dpi=300)
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Place title matching original matplotlib style
    fig.text(0.5, 0.99, new_title, ha="center", va="top",
             fontsize=14, fontweight="bold")
    fig.savefig(dst_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {dst_path}")


def main():
    for src_file, new_title, dst_file in TRUE_RECALL_FILES:
        src_path = os.path.join(SRC_DIR_RECALL, src_file)
        if not os.path.exists(src_path):
            print(f"  SKIP (not found): {src_path}")
            continue

        dst_path = os.path.join(DST_DIR, dst_file)
        retitle_image(src_path, dst_path, new_title)

    print(f"\nAll outputs in: {DST_DIR}")


if __name__ == "__main__":
    main()
