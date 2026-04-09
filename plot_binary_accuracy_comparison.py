# This code was written with the assistance of Claude (Anthropic).

import matplotlib.pyplot as plt
import numpy as np

# === BINARY ACCURACIES (from job outputs) ===
# 1-year: OrganizedDatasetBalancedBinary (~11k train, ~2.8k test)

models = [
    "U-Net",
    "InceptionV3",
    "VGG16",
    "AlexNet",
    "FirstCNN",
    "ResNet50",
]
acc = [68.01, 65.65, 64.37, 63.75, 63.44, 62.68]

# === PLOT ===
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
bars = ax.bar(x, acc, 0.6, color="#3498db")

ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("Binary Classification: Overall Accuracy by Model (1-year)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right")
ax.set_ylim(0, 80)
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

# Add value labels on bars
for bar in bars:
    ax.annotate(f"{bar.get_height():.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("/scratch/gpfs/ALAINK/Suthi/binary_accuracy_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: binary_accuracy_comparison.png")

# === VGG16 F1 SCORES ===
print("\n" + "=" * 50)
print("VGG16 BINARY (1-year, safe vs dangerous)")
print("=" * 50)
print("  dangerous:  precision 0.63, recall 0.69, F1 0.66")
print("  safe:       precision 0.66, recall 0.60, F1 0.63")
print("  macro avg F1:   0.64")
print("  weighted avg F1: 0.64")
print("  accuracy: 64.37%")

print("\n" + "=" * 50)
print("VGG16 MULTI-CLASS (1-year, 4 classes)")
print("=" * 50)
print("  0_safe:       precision 0.91, recall 0.51, F1 0.65")
print("  1_crash:      precision 0.17, recall 0.02, F1 0.04")
print("  2_crashes:    precision 0.04, recall 0.60, F1 0.08")
print("  3plus_crashes: precision 0.03, recall 0.40, F1 0.05")
print("  macro avg F1:   0.21")
print("  weighted avg F1: 0.55")
print("  accuracy: 44.51%")

# === InceptionV3 F1 SCORES ===
print("\n" + "=" * 50)
print("InceptionV3 BINARY (1-year, safe vs dangerous)")
print("=" * 50)
print("  dangerous:  precision 0.64, recall 0.70, F1 0.67")
print("  safe:       precision 0.67, recall 0.62, F1 0.64")
print("  macro avg F1:   0.66")
print("  weighted avg F1: 0.66")
print("  accuracy: 65.65%")

print("\n" + "=" * 50)
print("InceptionV3 MULTI-CLASS (1-year, 4 classes)")
print("=" * 50)
print("  0_safe:       precision 0.91, recall 0.55, F1 0.69")
print("  1_crash:      precision 0.14, recall 0.01, F1 0.02")
print("  2_crashes:    precision 0.04, recall 0.48, F1 0.07")
print("  3plus_crashes: precision 0.03, recall 0.59, F1 0.06")
print("  macro avg F1:   0.21")
print("  weighted avg F1: 0.57")
print("  accuracy: 47.55%")

# === FirstCNN (DIY CNN) F1 SCORES ===
print("\n" + "=" * 50)
print("FirstCNN (DIY CNN) BINARY (1-year, safe vs dangerous)")
print("=" * 50)
print("  dangerous:  precision 0.48, recall 0.07, F1 0.12")
print("  safe:       precision 0.64, recall 0.96, F1 0.77")
print("  macro avg F1:   0.44")
print("  weighted avg F1: 0.53")
print("  accuracy: 63.44%")

print("\n" + "=" * 50)
print("FirstCNN (DIY CNN) MULTI-CLASS (1-year, 4 classes)")
print("=" * 50)
print("  0_safe:       precision 0.84, recall 0.75, F1 0.79")
print("  1_crash:      precision 0.08, recall 0.00, F1 0.00")
print("  2_crashes:    precision 0.03, recall 0.02, F1 0.03")
print("  3plus_crashes: precision 0.01, recall 0.33, F1 0.03")
print("  macro avg F1:   0.21")
print("  weighted avg F1: 0.66")
print("  accuracy: 62.38%")

# Save VGG16, InceptionV3, and FirstCNN F1 to CSV
with open("/scratch/gpfs/ALAINK/Suthi/vgg16_inceptionv3_firstcnn_f1_scores.csv", "w") as f:
    f.write("model,class,precision,recall,f1\n")
    f.write("VGG16_binary,dangerous,0.63,0.69,0.66\n")
    f.write("VGG16_binary,safe,0.66,0.60,0.63\n")
    f.write("VGG16_binary,macro_avg,,,0.64\n")
    f.write("VGG16_binary,weighted_avg,,,0.64\n")
    f.write("VGG16_multi,0_safe,0.91,0.51,0.65\n")
    f.write("VGG16_multi,1_crash,0.17,0.02,0.04\n")
    f.write("VGG16_multi,2_crashes,0.04,0.60,0.08\n")
    f.write("VGG16_multi,3plus_crashes,0.03,0.40,0.05\n")
    f.write("VGG16_multi,macro_avg,,,0.21\n")
    f.write("VGG16_multi,weighted_avg,,,0.55\n")
    f.write("InceptionV3_binary,dangerous,0.64,0.70,0.67\n")
    f.write("InceptionV3_binary,safe,0.67,0.62,0.64\n")
    f.write("InceptionV3_binary,macro_avg,,,0.66\n")
    f.write("InceptionV3_binary,weighted_avg,,,0.66\n")
    f.write("InceptionV3_multi,0_safe,0.91,0.55,0.69\n")
    f.write("InceptionV3_multi,1_crash,0.14,0.01,0.02\n")
    f.write("InceptionV3_multi,2_crashes,0.04,0.48,0.07\n")
    f.write("InceptionV3_multi,3plus_crashes,0.03,0.59,0.06\n")
    f.write("InceptionV3_multi,macro_avg,,,0.21\n")
    f.write("InceptionV3_multi,weighted_avg,,,0.57\n")
    f.write("FirstCNN_binary,dangerous,0.48,0.07,0.12\n")
    f.write("FirstCNN_binary,safe,0.64,0.96,0.77\n")
    f.write("FirstCNN_binary,macro_avg,,,0.44\n")
    f.write("FirstCNN_binary,weighted_avg,,,0.53\n")
    f.write("FirstCNN_multi,0_safe,0.84,0.75,0.79\n")
    f.write("FirstCNN_multi,1_crash,0.08,0.00,0.00\n")
    f.write("FirstCNN_multi,2_crashes,0.03,0.02,0.03\n")
    f.write("FirstCNN_multi,3plus_crashes,0.01,0.33,0.03\n")
    f.write("FirstCNN_multi,macro_avg,,,0.21\n")
    f.write("FirstCNN_multi,weighted_avg,,,0.66\n")
print("\nSaved: vgg16_inceptionv3_firstcnn_f1_scores.csv")
print("\nDone!")
