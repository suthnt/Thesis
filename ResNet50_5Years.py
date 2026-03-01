"""ResNet50 Binary Classification - 5 YEARS DATA
Uses same train/test split as AlexNet 5Years (OrganizedDatasetBalancedBinary_5Years)
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
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

# ===================== CONFIGURATION - 5 YEARS =====================
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary_5Years"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

NUM_CLASSES = len([d for d in os.listdir(f"{DATASET_DIR}/train") if os.path.isdir(f"{DATASET_DIR}/train/{d}")])
print(f"=== ResNet50 5 Years ===")
print(f"Dataset: {DATASET_DIR}")
print(f"Detected {NUM_CLASSES} classes")

# ===================== DATA LOADING =====================
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

# ===================== MODEL =====================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ===================== TRAINING =====================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_resnet50_model_5Years.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
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

with open('resnet50_results_5Years.txt', 'w') as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Accuracy (%): {test_acc*100:.2f}%\n")

# ===================== CONFUSION MATRIX =====================
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_names = list(train_generator.class_indices.keys())

print("\n=== Classification Report (5 Years) ===")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
with open('resnet50_results_5Years.txt', 'a') as f:
    f.write("\n=== Classification Report ===\n")
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet50 5 Years - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_resnet50_5Years.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix_resnet50_5Years.png")

# ===================== TRAINING CURVES =====================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('ResNet50 5 Years - Loss')
plt.xlabel('Epoch')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('ResNet50 5 Years - Accuracy')
plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig('training_curves_resnet50_5Years.png', dpi=300, bbox_inches='tight')
print("Saved: training_curves_resnet50_5Years.png")

model.save("ResNet50_bin_5Years.keras")
print("Saved: ResNet50_bin_5Years.keras")
