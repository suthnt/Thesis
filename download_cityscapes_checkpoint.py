#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import urllib.request

URL = "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R50_bs16_90k/model_final_cc1b1f.pkl"
OUT = "/scratch/gpfs/ALAINK/Suthi/Mask2Former/checkpoints/maskformer2_R50_Cityscapes_semantic.pkl"
MIN_SIZE = 80_000_000

os.makedirs(os.path.dirname(OUT), exist_ok=True)

if os.path.isfile(OUT) and os.path.getsize(OUT) >= MIN_SIZE:
    print(f"Already have valid checkpoint: {OUT}")
    exit(0)

print("Downloading Mask2Former Cityscapes model (~100MB)...")
try:
    urllib.request.urlretrieve(URL, OUT)
except Exception as e:
    print(f"Download failed: {e}")
    if os.path.isfile(OUT):
        os.remove(OUT)
    exit(1)

sz = os.path.getsize(OUT)
if sz >= MIN_SIZE:
    print(f"OK: {OUT} ({sz / 1e6:.1f} MB)")
    print("Now run: sbatch slurm_scripts/mask2former_features.slurm")
else:
    print(f"ERROR: File too small ({sz} bytes). Delete and retry.")
    os.remove(OUT)
    exit(1)
