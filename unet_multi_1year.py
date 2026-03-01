"""
U-Net Multi-Class - 1-YEAR DATA
Classes: 0_safe, 1_crash, 2_crashes, 3plus_crashes
Same dataset as AlexNet_multi_1year. Handles severe imbalance (~84% safe)
with aggressive class weights.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    GlobalAveragePooling2D, Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClass"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 80
MODEL_TAG = "unet_multi_1year"

NUM_CLASSES = len([d for d in os.listdir(f"{DATASET_DIR}/train") if os.path.isdir(f"{DATASET_DIR}/train/{d}")])
print(f"=== {MODEL_TAG} ===")
print(f"Dataset: {DATASET_DIR}")
print(f"Classes: {NUM_CLASSES}")

# === DATA ===
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)
test_generator = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# Print class distribution
unique, counts = np.unique(train_generator.classes, return_counts=True)
total = len(train_generator.classes)
print(f"\nClass distribution (train):")
for cls, cnt in zip(unique, counts):
    pct = 100 * cnt / total
    print(f"  class {cls}: {cnt} ({pct:.1f}%)")
print(f"Total: {total}")

# Aggressive class weights - boost minority classes 2x (same as AlexNet multi)
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)
class_weights_dict = {}
for i, w in enumerate(class_weights):
    pct = 100 * counts[i] / total
    if pct < 5:
        class_weights_dict[i] = float(w) * 2.0
    else:
        class_weights_dict[i] = float(w)
print(f"Class weights (minority boosted): {class_weights_dict}")

# === U-NET ENCODER + CLASSIFICATION HEAD ===
def conv_block(x, filters, name_prefix):
    x = Conv2D(filters, 3, padding="same", activation="relu", name=f"{name_prefix}_conv1")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = Conv2D(filters, 3, padding="same", activation="relu", name=f"{name_prefix}_conv2")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn2")(x)
    return x


def build_unet_classifier(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=4):
    inputs = Input(shape=input_shape)
    c1 = conv_block(inputs, 32, "enc1")
    p1 = MaxPooling2D(2, name="pool1")(c1)
    p1 = Dropout(0.1)(p1)
    c2 = conv_block(p1, 64, "enc2")
    p2 = MaxPooling2D(2, name="pool2")(c2)
    p2 = Dropout(0.1)(p2)
    c3 = conv_block(p2, 128, "enc3")
    p3 = MaxPooling2D(2, name="pool3")(c3)
    p3 = Dropout(0.2)(p3)
    c4 = conv_block(p3, 256, "enc4")
    p4 = MaxPooling2D(2, name="pool4")(c4)
    p4 = Dropout(0.2)(p4)
    bottleneck = conv_block(p4, 512, "bottleneck")
    bottleneck = Dropout(0.3)(bottleneck)
    x = GlobalAveragePooling2D(name="gap")(bottleneck)
    x = Dense(256, activation="relu", name="fc1")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", name="output")(x)
    return Model(inputs, outputs, name="unet_classifier")


model = build_unet_classifier(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# === TRAIN ===
callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
    ModelCheckpoint(f"best_{MODEL_TAG}.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks,
)

# === EVALUATE ===
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

test_generator.reset()
y_pred = np.argmax(model.predict(test_generator), axis=1)
y_true = test_generator.classes
class_names = list(train_generator.class_indices.keys())

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
fig_size = max(8, NUM_CLASSES * 2)
plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"{MODEL_TAG} Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"confusion_matrix_{MODEL_TAG}.png", dpi=300, bbox_inches="tight")
print(f"Saved: confusion_matrix_{MODEL_TAG}.png")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.legend()
plt.title("Accuracy")
plt.tight_layout()
plt.savefig(f"training_curves_{MODEL_TAG}.png", dpi=300, bbox_inches="tight")
print(f"Saved: training_curves_{MODEL_TAG}.png")

model.save(f"{MODEL_TAG}.keras")
print(f"Saved: {MODEL_TAG}.keras")
