#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

MODEL = "facebook/mask2former-swin-small-ade-semantic"
print(f"Downloading {MODEL} to cache (run on login node)...")
AutoImageProcessor.from_pretrained(MODEL)
Mask2FormerForUniversalSegmentation.from_pretrained(MODEL)
print("Done. Model cached. Now run: sbatch slurm_scripts/mask2former_features_hf.slurm")
