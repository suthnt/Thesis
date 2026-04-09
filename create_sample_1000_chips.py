#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import argparse
import os
import random
import shutil
import zipfile

SOURCE = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClass"
CLASSES = ["0_safe", "1_crash", "2_crashes", "3plus_crashes"]
EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="Total samples (default 1000)")
    ap.add_argument("--output", "-o", default="sample_1000_chips_1year", help="Output directory")
    ap.add_argument("--zip", action="store_true", help="Create zip archive for download")
    args = ap.parse_args()

    n_per_class = args.n // len(CLASSES)  # 250 each for 1000
    out_dir = os.path.join("/scratch/gpfs/ALAINK/Suthi", args.output)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for cls in CLASSES:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    print(f"Sampling {n_per_class} per class from 1-year multiclass chips...")
    random.seed(42)

    for cls in CLASSES:
        files = []
        for split in ["train", "test"]:
            folder = os.path.join(SOURCE, split, cls)
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(EXT):
                        files.append(os.path.join(folder, f))

        chosen = random.sample(files, min(n_per_class, len(files)))
        for src in chosen:
            dst = os.path.join(out_dir, cls, os.path.basename(src))
            shutil.copy2(src, dst)
        print(f"  {cls}: {len(chosen)} images")

    total = sum(len(os.listdir(os.path.join(out_dir, c))) for c in CLASSES)
    print(f"\nTotal: {total} images in {out_dir}")

    if args.zip:
        zip_path = out_dir + ".zip"
        print(f"Creating {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, fnames in os.walk(out_dir):
                for f in fnames:
                    path = os.path.join(root, f)
                    zf.write(path, os.path.relpath(path, os.path.dirname(out_dir)))
        print(f"Done. Download with: scp <user>@della.princeton.edu:{zip_path} .")
    else:
        print(f"\nTo create a zip for download: python create_sample_1000_chips.py --zip")


if __name__ == "__main__":
    main()
