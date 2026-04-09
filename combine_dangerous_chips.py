#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import shutil

base = "/scratch/gpfs/ALAINK/Suthi/ChippedImages_5Years"
dangerous_dir = os.path.join(base, "dangerous")
os.makedirs(dangerous_dir, exist_ok=True)

dangerous_sources = ["1_crash", "2_crashes", "3plus_crashes"]
total = 0

for source in dangerous_sources:
    src_dir = os.path.join(base, source)
    if not os.path.exists(src_dir):
        print(f"  Skipping {source} (not found)")
        continue
    count = 0
    for f in os.listdir(src_dir):
        if f.endswith(".tif"):
            # Prefix with source to avoid name collisions (each folder has chip_000000, etc.)
            new_name = f"{source}_{f}"
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dangerous_dir, new_name))
            count += 1
    print(f"{source}: copied {count} chips")
    total += count

print(f"\nDone! Total: {total} chips in {dangerous_dir}")
