# This code was written with the assistance of Claude (Anthropic).

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Config
MODEL_PATH = "/scratch/gpfs/ALAINK/Suthi/ResNet50_4class.keras"
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetMultiClass"
IMG_SIZE = 224
BATCH_SIZE = 32

# Load model
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Test samples: {test_generator.samples}")
print(f"Classes: {test_generator.class_indices}")

# Get predictions
print("Running predictions...")
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

# Classification report
print("\n=== Classification Report ===")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\n=== Confusion Matrix (raw numbers) ===")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet50 Multi-Class Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_resnet50_multi.png', dpi=300, bbox_inches='tight')
print("\nSaved: confusion_matrix_resnet50_multi.png")

# Also save normalized version
plt.figure(figsize=(10, 8))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet50 Multi-Class Confusion Matrix (Normalized)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_resnet50_multi_normalized.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix_resnet50_multi_normalized.png")

