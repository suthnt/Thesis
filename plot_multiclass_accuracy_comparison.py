# This code was written with the assistance of Claude (Anthropic).

import matplotlib.pyplot as plt
import numpy as np

# === MULTI-CLASS ACCURACIES (from job outputs) ===
# 1-year: OrganizedDatasetMultiClass (4 classes: 0_safe, 1_crash, 2_crashes, 3plus_crashes)

models = [
    "U-Net",
    "FirstCNN",
    "AlexNet",
    "InceptionV3",
    "VGG16",
]
acc = [75.74, 62.38, 61.21, 47.55, 44.51]

# === PLOT ===
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
bars = ax.bar(x, acc, 0.6, color="#9b59b6")

ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("Multi-Class Classification: Overall Accuracy by Model (1-year)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right")
ax.set_ylim(0, 90)
ax.axhline(y=25, color="gray", linestyle="--", alpha=0.5)  # random guess for 4 classes

# Add value labels on bars
for bar in bars:
    ax.annotate(f"{bar.get_height():.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("/scratch/gpfs/ALAINK/Suthi/multiclass_accuracy_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: multiclass_accuracy_comparison.png")

# === U-Net MULTI-CLASS F1 SCORES ===
print("\n" + "=" * 50)
print("U-Net MULTI-CLASS (1-year, 4 classes)")
print("=" * 50)
print("  0_safe:       precision 0.84, recall 0.91, F1 0.87")
print("  1_crash:      precision 0.17, recall 0.00, F1 0.00")
print("  2_crashes:    precision 0.04, recall 0.03, F1 0.03")
print("  3plus_crashes: precision 0.03, recall 0.21, F1 0.04")
print("  macro avg F1:   0.24")
print("  weighted avg F1: 0.73")
print("  accuracy: 75.74%")

# Save U-Net multi F1 to CSV
with open("/scratch/gpfs/ALAINK/Suthi/unet_multi_f1_scores.csv", "w") as f:
    f.write("model,class,precision,recall,f1\n")
    f.write("U-Net_multi,0_safe,0.84,0.91,0.87\n")
    f.write("U-Net_multi,1_crash,0.17,0.00,0.00\n")
    f.write("U-Net_multi,2_crashes,0.04,0.03,0.03\n")
    f.write("U-Net_multi,3plus_crashes,0.03,0.21,0.04\n")
    f.write("U-Net_multi,macro_avg,,,0.24\n")
    f.write("U-Net_multi,weighted_avg,,,0.73\n")
print("\nSaved: unet_multi_f1_scores.csv")
print("\nDone!")
