#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os

WEIGHTS_DIR = "/scratch/gpfs/ALAINK/Suthi/keras_pretrained_weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# URLs from Keras applications
URLS = {
    "vgg16": "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
    "inception_v3": "https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
}

for name, url in URLS.items():
    path = os.path.join(WEIGHTS_DIR, os.path.basename(url))
    if os.path.isfile(path) and os.path.getsize(path) > 10_000_000:
        print(f"{name}: already have {path}")
        continue
    print(f"Downloading {name}...")
    import urllib.request
    urllib.request.urlretrieve(url, path)
    print(f"  Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)")

print("\nDone. Now run: sbatch slurm_scripts/VGG16_1year.slurm")
print("              sbatch slurm_scripts/InceptionV3_1year.slurm")
