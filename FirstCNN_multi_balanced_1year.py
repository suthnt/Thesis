"""
FirstCNN Multi-Class - BALANCED 1-YEAR DATA
Uses OrganizedDatasetMultiClassBalanced (equal samples per class + 90/180/270 rotations).
All outputs saved to balanced_multiclass_1year/
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClassBalanced"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/balanced_multiclass_1year"
IMG_SIZE = 99
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 50
MODEL_TAG = "FirstCNN_multi_balanced_1year"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"=== {MODEL_TAG} ===")
print(f"Dataset: {DATASET_DIR}")
print(f"Output: {OUTPUT_DIR}")

# === DATA ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)
test_generator = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

print("Class indices:", train_generator.class_indices)
print(f"Training samples: {train_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# === MODEL ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === TRAIN ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, f'best_{MODEL_TAG}.keras'), monitor='val_accuracy', save_best_only=True, verbose=1),
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
)

# === EVALUATE ===
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

test_generator.reset()
y_pred = np.argmax(model.predict(test_generator), axis=1)
y_true = test_generator.classes
class_names = list(train_generator.class_indices.keys())

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title(f'{MODEL_TAG} - Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title(f'{MODEL_TAG} - Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'training_history_{MODEL_TAG}.png'), dpi=300, bbox_inches='tight')
print(f"Saved: training_history_{MODEL_TAG}.png")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'{MODEL_TAG} - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{MODEL_TAG}.png'), dpi=300, bbox_inches='tight')
print(f"Saved: confusion_matrix_{MODEL_TAG}.png")

model.save(os.path.join(OUTPUT_DIR, f"{MODEL_TAG}.keras"))
print(f"Saved: {MODEL_TAG}.keras")
print(f"\nAll outputs in: {OUTPUT_DIR}")
