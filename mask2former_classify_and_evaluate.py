#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

FEATURES_DIR = "/scratch/gpfs/ALAINK/Suthi/mask2former_features_1year_cityscapes"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/mask2former_eval_1year_cityscapes"
CLASS_NAMES = ["safe", "dangerous"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load features
print("Loading Mask2Former features...")
train_data = np.load(os.path.join(FEATURES_DIR, "train_features.npz"))
test_data = np.load(os.path.join(FEATURES_DIR, "test_features.npz"))

X_train = train_data["features"]
y_train = train_data["labels"]
X_test = test_data["features"]
y_test = test_data["labels"]

print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]} samples")

# Train classifier
print("\nTraining LogisticRegression...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Metrics for the bar chart
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
metrics = {
    "Accuracy": accuracy,
    "Precision (macro)": precision,
    "Recall (macro)": recall,
    "F1 (macro)": f1,
}

# --- Plot 1: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=ax,
    cbar_kws={"label": "Count"},
)
ax.set_title("Mask2Former (Cityscapes) + LogisticRegression\nConfusion Matrix (1-Year Dataset)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/confusion_matrix.png")

# --- Plot 2: Overall Accuracy & Metrics Bar Chart ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
bars = ax.bar(metrics.keys(), list(metrics.values()), color=colors[: len(metrics)], edgecolor="white")

# Add value labels on bars
for bar, val in zip(bars, metrics.values()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.1%}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

ax.set_ylabel("Score")
ax.set_title("Mask2Former (Cityscapes) + LogisticRegression\nOverall Performance (1-Year Dataset)")
ax.set_ylim(0, 1.15)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "accuracy_metrics.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/accuracy_metrics.png")

# --- Plot 3: Normalized confusion matrix ---
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".1%",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=ax,
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Fraction"},
)
ax.set_title("Mask2Former (Cityscapes) + LogisticRegression\nConfusion Matrix (Normalized)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/confusion_matrix_normalized.png")

print(f"\n=== Done. All outputs in {OUTPUT_DIR} ===")
print(f"Overall Accuracy: {accuracy:.2%}")
