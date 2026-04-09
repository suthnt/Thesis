#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import glob
import argparse
import csv
from pathlib import Path

# === CONFIG ===
DEFAULT_IMAGES_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary/test"
DEFAULT_LABELS = ["bike_lane", "building_shadow", "person_visible", "none"]
OUTPUT_CSV = "/scratch/gpfs/ALAINK/Suthi/tagging_labels.csv"
IMG_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif")


def get_image_paths(images_dir):
    """Collect images from directory (flat or safe/dangerous subdirs)."""
    paths = []
    for ext in IMG_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    # Also check subdirs (safe, dangerous, etc.)
    for sub in ["safe", "dangerous", "0_safe", "1_crash", "2_crashes", "3plus_crashes"]:
        subdir = os.path.join(images_dir, sub)
        if os.path.isdir(subdir):
            for ext in IMG_EXTENSIONS:
                paths.extend(glob.glob(os.path.join(subdir, ext)))
    return sorted(set(paths))


def load_existing_tags(csv_path, labels):
    """Load already-tagged filenames and their labels."""
    tags = {}
    if os.path.isfile(csv_path):
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                fn = row.get("filename", "")
                lidx = row.get("label_idx", "")
                lbl = row.get("label", "")
                if lidx != "" and lidx.isdigit():
                    tags[fn] = int(lidx)
                elif lbl:
                    try:
                        tags[fn] = labels.index(lbl)
                    except ValueError:
                        tags[fn] = lbl
                else:
                    tags[fn] = lbl
    return tags


def save_tags(csv_path, tags, labels):
    """Save tags to CSV."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "label_idx"])
        for fn, val in sorted(tags.items()):
            if isinstance(val, int):
                lbl = labels[val] if 0 <= val < len(labels) else ""
                w.writerow([fn, lbl, val])
            else:
                w.writerow([fn, val, ""])
    print(f"Saved {len(tags)} tags to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple image tagging for concepts")
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing images (or with safe/dangerous subdirs)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels, e.g. bike_lane,shadow,person",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_CSV,
        help="Output CSV path",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index to start from (for resuming)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max images to show (0 = no limit)",
    )
    parser.add_argument(
        "--skip-tagged",
        action="store_true",
        help="Skip images that already have a label",
    )
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not labels:
        labels = DEFAULT_LABELS

    paths = get_image_paths(args.images_dir)
    if not paths:
        print(f"No images found in {args.images_dir}")
        return

    existing = load_existing_tags(args.output, labels)
    if args.skip_tagged:
        paths = [p for p in paths if os.path.basename(p) not in existing]
    if not paths:
        print("All images already tagged.")
        return

    # Apply start and limit
    start = min(args.start, len(paths) - 1)
    paths = paths[start:]
    if args.limit > 0:
        paths = paths[: args.limit]

    print(f"=== Simple Tagging ===")
    print(f"Images dir: {args.images_dir}")
    print(f"Labels: {labels}")
    print(f"  Keys: 1={labels[0]}, 2={labels[1] if len(labels)>1 else '-'}, ... n=skip, b=back, q=quit")
    print(f"Images to tag: {len(paths)}")
    print(f"Output: {args.output}")
    print()

    try:
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
    except ImportError:
        print("Need matplotlib. Install: pip install matplotlib")
        return

    tags = load_existing_tags(args.output, labels)

    idx = 0
    n = len(paths)

    def on_key(event):
        nonlocal idx
        key = event.key
        if key == "q":
            plt.close()
            return
        fn = os.path.basename(paths[idx])
        if key == "n":
            idx_new = min(idx + 1, n - 1)
            if idx_new != idx:
                idx = idx_new
                show_image()
            return
        if key == "b":
            idx_new = max(idx - 1, 0)
            if idx_new != idx:
                idx = idx_new
                show_image()
            return
        num = key
        if num.isdigit():
            label_idx = int(num) - 1
            if 0 <= label_idx < len(labels):
                tags[fn] = label_idx
                print(f"  Tagged {fn} -> {labels[label_idx]}")
                idx = min(idx + 1, n - 1)
                if idx == n - 1 and paths[idx] == paths[-1]:
                    show_image()
                    return
                show_image()

    def show_image():
        if idx >= n:
            plt.close()
            return
        path = paths[idx]
        fn = os.path.basename(path)
        img = imread(path)
        ax.clear()
        ax.imshow(img)
        current_tag = tags.get(fn, "")
        if isinstance(current_tag, int) and 0 <= current_tag < len(labels):
            current_tag = labels[current_tag]
        ax.set_title(f"[{idx+1}/{n}] {fn} | Tagged: {current_tag or '-'}")
        ax.axis("off")
        fig.canvas.draw()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.mpl_connect("key_press_event", on_key)
    show_image()
    plt.show()

    save_tags(args.output, tags, labels)
    print("Done.")


if __name__ == "__main__":
    main()
