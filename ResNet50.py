# This code was written with the assistance of Claude (Anthropic).

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===================== CONFIGURATION =====================
BASE = "/scratch/gpfs/ALAINK/Suthi"
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=None, help="K-fold index 0..4 (None = standard train/test)")
args = parser.parse_args()

if args.fold is not None:
    DATASET_DIR = f"{BASE}/OrganizedDatasetBalancedBinary_kfold/fold_{args.fold}"
    OUTPUT_DIR = f"{BASE}/binary_1year_kfold/ResNet50_bin_f{args.fold}"
    VAL_SPLIT = "val"
else:
    DATASET_DIR = f"{BASE}/OrganizedDatasetBalancedBinary"
    OUTPUT_DIR = BASE
    VAL_SPLIT = "test"

IMG_SIZE = 224  # ResNet50 expects 224x224 images
BATCH_SIZE = 32
EPOCHS = 50

# ===================== DATA LOADING =====================
# Auto-detect number of classes
NUM_CLASSES = len([d for d in os.listdir(f"{DATASET_DIR}/train") if os.path.isdir(f"{DATASET_DIR}/train/{d}")])
print(f"Detected {NUM_CLASSES} classes")

# Data generators
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
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/{VAL_SPLIT}",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Class indices:", train_generator.class_indices)
print(f"Training samples: {train_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# Calculate class weights for imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weights_dict}")

# ===================== MODEL BUILDING =====================
base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# ===================== TRAINING =====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_resnet50_model.keras'), monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ===================== EVALUATION =====================
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

if args.fold is not None:
    import json
    with open(os.path.join(OUTPUT_DIR, "kfold_result.json"), "w") as f:
        json.dump({"fold": args.fold, "val_accuracy": float(test_acc)}, f)

# Save test accuracy to file
res_path = os.path.join(OUTPUT_DIR, 'resnet50_results.txt')
with open(res_path, 'w') as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Accuracy (%): {test_acc*100:.2f}%\n")
print("Saved: resnet50_results.txt")

# ===================== CONFUSION MATRIX =====================
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

class_names = list(train_generator.class_indices.keys())

# Classification report
print("\n=== Classification Report ===")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Save classification report to file
with open(res_path, 'a') as f:
    f.write("\n=== Classification Report ===\n")
    f.write(report)

# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
fig_size = max(8, NUM_CLASSES * 1.5)
plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet50 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_resnet50.png'), dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix_resnet50.png")

# ===================== TRAINING CURVES =====================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves_resnet50.png'), dpi=300, bbox_inches='tight')
print("Saved: training_curves_resnet50.png")

# Save model
model.save(os.path.join(OUTPUT_DIR, f"ResNet50_{NUM_CLASSES}class.keras"))
print(f"Saved: ResNet50_{NUM_CLASSES}class.keras")
