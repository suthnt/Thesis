"""FirstCNN (DIY CNN) Binary Classification - 5 YEARS DATA
Uses same train/test split as AlexNet 5Years (OrganizedDatasetBalancedBinary_5Years)
"""

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

# === CONFIGURATION - 5 YEARS ===
MODEL_TAG = "FirstCNN_bin_5Years"
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary_5Years"
IMG_SIZE = 99
BATCH_SIZE = 32
NUM_CLASSES = 2
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

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === TRAINING ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_FirstCNN_bin_5Years.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# === EVALUATION ===
print("\n=== Final Evaluation (5 Years) ===")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")

# === PLOTS ===
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
plt.savefig('training_history_FirstCNN_bin_5Years.png', dpi=300, bbox_inches='tight')
print("Saved: training_history_FirstCNN_bin_5Years.png")

# Confusion Matrix
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_names = list(train_generator.class_indices.keys())

print("\n=== Classification Report (5 Years) ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'{MODEL_TAG} - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_FirstCNN_bin_5Years.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix_FirstCNN_bin_5Years.png")

# Sample predictions
OUTPUT_DIR = '/scratch/gpfs/ALAINK/Suthi/samples_FirstCNN_bin_5Years'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sample_gen = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=6,
    class_mode='categorical',
    shuffle=True
)
images, labels = next(sample_gen)
preds = model.predict(images)
for i in range(len(images)):
    true_label = class_names[np.argmax(labels[i])]
    pred_label = class_names[np.argmax(preds[i])]
    confidence = preds[i][np.argmax(preds[i])] * 100
    plt.figure(figsize=(4, 4))
    plt.imshow(images[i])
    plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)")
    plt.axis('off')
    plt.savefig(f"{OUTPUT_DIR}/sample_{i+1}_{true_label}.png", bbox_inches='tight', dpi=150)
    plt.close()
print(f"Saved samples to {OUTPUT_DIR}/")

model.save("FirstCNN_bin_5Years.keras")
print("Saved: FirstCNN_bin_5Years.keras")
