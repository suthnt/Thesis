# This code was written with the assistance of Claude (Anthropic).

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === CONFIG ===
MODEL_PATH = "/scratch/gpfs/ALAINK/Suthi/best_unet_1year.keras"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/unet_feature_importance"
NPZ_DIR = "/scratch/gpfs/ALAINK/Suthi/unet_features_1year"
TOP_N = 30  # Top/bottom features to show

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("=== U-Net Feature Importance ===")

# === 1. Classifier weights (direct from model) ===
model = load_model(MODEL_PATH)
output_layer = model.get_layer("output")
W, b = output_layer.get_weights()
# W shape (256, 2): W[:,0]=safe, W[:,1]=dangerous
# Importance: positive = contributes to dangerous, negative = contributes to safe
importance = W[:, 1] - W[:, 0]
sorted_idx = np.argsort(importance)[::-1]

# Plot: top and bottom features
fig, ax = plt.subplots(figsize=(10, 10))
n_show = min(TOP_N, len(importance) // 2)
top_idx = sorted_idx[:n_show]
bot_idx = sorted_idx[-n_show:][::-1]
all_idx = np.concatenate([top_idx, bot_idx])
labels = [f"feat_{i}" for i in all_idx]
vals = importance[all_idx]
colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
ax.barh(range(len(all_idx)), vals, color=colors)
ax.set_yticks(range(len(all_idx)))
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Classifier weight (dangerous − safe)\npositive → predicts dangerous")
ax.set_title("U-Net: Which fc1 features drive the classifier?")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "unet_classifier_weights.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/unet_classifier_weights.png")

# CSV: all 256 features ranked
csv_path = os.path.join(OUTPUT_DIR, "unet_classifier_weights.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["feature_idx", "weight_dangerous_minus_safe", "rank"])
    for r, i in enumerate(sorted_idx):
        w.writerow([i, f"{importance[i]:.6f}", r + 1])
print(f"Saved: {csv_path}")

# === 2. Correlation with labels (if extracted features exist) ===
train_npz = os.path.join(NPZ_DIR, "train_features.npz")
test_npz = os.path.join(NPZ_DIR, "test_features.npz")
if os.path.isfile(train_npz) and os.path.isfile(test_npz):
    train = np.load(train_npz)
    test = np.load(test_npz)
    X_all = np.vstack([train["features"], test["features"]])
    y_all = np.concatenate([train["labels"], test["labels"]])
    corr = np.array([
        np.corrcoef(X_all[:, i], y_all)[0, 1] if np.isfinite(X_all[:, i]).all() else 0
        for i in range(X_all.shape[1])
    ])
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr_sorted = np.argsort(np.abs(corr))[::-1]

    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Left: classifier weights (top 20)
    ax = axes[0]
    idx20 = sorted_idx[:20]
    ax.barh(range(20), importance[idx20], color=["#e74c3c" if v > 0 else "#3498db" for v in importance[idx20]])
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"feat_{i}" for i in idx20], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Classifier weight")
    ax.set_title("Top 20 features by classifier weight")

    # Right: correlation with dangerous label
    ax = axes[1]
    idx20c = corr_sorted[:20]
    ax.barh(range(20), corr[idx20c], color=["#e74c3c" if c > 0 else "#3498db" for c in corr[idx20c]])
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"feat_{i}" for i in idx20c], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Correlation with dangerous")
    ax.set_title("Top 20 features by label correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "unet_feature_importance_dual.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/unet_feature_importance_dual.png")

    csv2 = os.path.join(OUTPUT_DIR, "unet_feature_correlation.csv")
    with open(csv2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_idx", "correlation_with_dangerous"])
        for i in corr_sorted:
            w.writerow([i, f"{corr[i]:.4f}"])
    print(f"Saved: {csv2}")
else:
    print(f"Skipping correlation (run unet_feature_extraction.py first for {NPZ_DIR})")

print("\n=== Done ===")
