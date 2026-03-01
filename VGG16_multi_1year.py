"""
VGG16 Multi-Class - 1-YEAR DATA
Classes: 0_safe, 1_crash, 2_crashes, 3plus_crashes
Uses OrganizedDatasetMultiClass. Handles severe imbalance (~84% safe) with aggressive class weights.
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClass"
WEIGHTS_DIR = "/scratch/gpfs/ALAINK/Suthi/keras_pretrained_weights"
VGG16_WEIGHTS = os.path.join(WEIGHTS_DIR, "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 80
MODEL_TAG = "VGG16_multi_1year"

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

unique, counts = np.unique(train_generator.classes, return_counts=True)
total = len(train_generator.classes)
print(f"\nClass distribution (train):")
for cls, cnt in zip(unique, counts):
    pct = 100 * cnt / total
    print(f"  class {cls}: {cnt} ({pct:.1f}%)")
print(f"Total: {total}")

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)
class_weights_dict = {}
for i, w in enumerate(class_weights):
    pct = 100 * counts[i] / total
    class_weights_dict[i] = float(w) * 2.0 if pct < 5 else float(w)
print(f"Class weights (minority boosted): {class_weights_dict}")

# === MODEL ===
if not os.path.isfile(VGG16_WEIGHTS):
    raise FileNotFoundError(
        f"VGG16 weights not found at {VGG16_WEIGHTS}. "
        "Run on LOGIN NODE: python download_keras_weights.py"
    )
base_model = VGG16(weights=VGG16_WEIGHTS, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === TRAIN ===
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
