"""
AlexNet Multi-Class - 1-YEAR DATA
Classes: 0_safe, 1_crash, 2_crashes, 3plus_crashes
Handles severe imbalance (~84% safe) with aggressive class weights + lower dropout.
"""

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClass"
IMG_SIZE = 227
BATCH_SIZE = 32
EPOCHS = 80
MODEL_TAG = "AlexNet_multi_1year"

NUM_CLASSES = len([d for d in os.listdir(f"{DATASET_DIR}/train") if os.path.isdir(f"{DATASET_DIR}/train/{d}")])
print(f"=== {MODEL_TAG} ===")
print(f"Dataset: {DATASET_DIR}")
print(f"Classes: {NUM_CLASSES}")

# === DATA ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
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

# Print class distribution
unique, counts = np.unique(train_generator.classes, return_counts=True)
total = len(train_generator.classes)
print(f"\nClass distribution (train):")
for cls, cnt in zip(unique, counts):
    pct = 100 * cnt / total
    print(f"  class {cls}: {cnt} ({pct:.1f}%)")
print(f"Total: {total}")

# AGGRESSIVE class weights - boost minority classes 2x to counteract ~84% safe
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)
# Extra boost for rare classes (< 5% of data)
class_weights_dict = {}
for i, w in enumerate(class_weights):
    pct = 100 * counts[i] / total
    if pct < 5:
        class_weights_dict[i] = float(w) * 2.0  # double weight for rare classes
    else:
        class_weights_dict[i] = float(w)
print(f"Class weights (minority boosted): {class_weights_dict}")

# === MODEL ===
# Lower dropout (0.3) - 0.5 was too aggressive for learning minority classes
model = tf.keras.Sequential([
    Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.3),
    Dense(4096, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === TRAIN ===
# More patience - minority classes need more epochs to learn
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
    ModelCheckpoint(f'best_{MODEL_TAG}.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
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

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig_size = max(8, NUM_CLASSES * 2)
plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'{MODEL_TAG} - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(f'confusion_matrix_{MODEL_TAG}.png', dpi=300, bbox_inches='tight')
print(f"Saved: confusion_matrix_{MODEL_TAG}.png")

# Training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title('Accuracy')
plt.tight_layout()
plt.savefig(f'training_curves_{MODEL_TAG}.png', dpi=300, bbox_inches='tight')
print(f"Saved: training_curves_{MODEL_TAG}.png")

model.save(f'{MODEL_TAG}.keras')
print(f"Saved: {MODEL_TAG}.keras")
