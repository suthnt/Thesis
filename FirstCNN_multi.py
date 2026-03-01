"""FirstCNN - Multi-Class Classification (0_safe, 1_crash, 2_crashes, 3plus_crashes)"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIGURATION ===
MODEL_TAG = "FirstCNN_multi"
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClass"
IMG_SIZE = 99
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 50

print(f"=== {MODEL_TAG} ===")
print(f"Dataset: {DATASET_DIR}")
print(f"Classes: {NUM_CLASSES}")

# === DATA LOADING ===
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
    f"{DATASET_DIR}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Class indices:", train_generator.class_indices)
print(f"Training samples: {train_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# Class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weights_dict}")

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
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === TRAINING ===
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        f'best_{MODEL_TAG}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# === EVALUATION ===
print("\n=== Final Evaluation ===")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")

# === PLOTS ===
# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title(f'{MODEL_TAG} - Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title(f'{MODEL_TAG} - Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'training_history_{MODEL_TAG}.png', dpi=300, bbox_inches='tight')
print(f"Saved: training_history_{MODEL_TAG}.png")

# Confusion Matrix
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

class_names = list(train_generator.class_indices.keys())

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'{MODEL_TAG} - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(f'confusion_matrix_{MODEL_TAG}.png', dpi=300, bbox_inches='tight')
print(f"Saved: confusion_matrix_{MODEL_TAG}.png")

# === SAMPLE PREDICTIONS ===
OUTPUT_DIR = f'/scratch/gpfs/ALAINK/Suthi/samples_{MODEL_TAG}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sample_gen = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=6,
    class_mode='categorical',
    shuffle=True
)
images, labels = next(sample_gen)
predictions = model.predict(images)

print("\n=== Sample Predictions ===")
for i in range(len(images)):
    true_idx = np.argmax(labels[i])
    pred_idx = np.argmax(predictions[i])
    true_label = class_names[true_idx]
    pred_label = class_names[pred_idx]
    confidence = predictions[i][pred_idx] * 100
    correct = "✓" if true_idx == pred_idx else "✗"
    
    print(f"Image {i+1}: True={true_label}, Predicted={pred_label} ({confidence:.1f}%) {correct}")
    
    filename = f"{OUTPUT_DIR}/sample_{i+1}_{true_label}.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(images[i])
    plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)")
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

print(f"\nSaved {len(images)} sample images to {OUTPUT_DIR}/")

# Save final model
model.save(f"{MODEL_TAG}.keras")
print(f"Saved: {MODEL_TAG}.keras")
