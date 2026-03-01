import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# === UPDATE THIS PATH ===
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBinary"  # Your organized folder

# Image settings
IMG_SIZE = 99  # Your images are 99x99
BATCH_SIZE = 32
NUM_CLASSES = 2 # Change depending on binary or multi class

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to 0-1
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,       # Intersections can be viewed from any direction
)

# Only rescaling for test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data from folders
train_generator = train_datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Load test data from folders
test_generator = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Print class mapping
print("Class indices:", train_generator.class_indices)
print(f"Training samples: {train_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weights_dict}") #keep goin


# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Model for 99x99 RGB images, 4 classes
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Block 4
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Classifier
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')  # 4 classes
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy',  # Use categorical for one-hot encoded labels
    metrics=['accuracy']
)

model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Callbacks to prevent overfitting and save best model
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_intersection_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    class_weight=class_weights_dict,  # Handle class imbalance
    callbacks=callbacks
)

# Evaluate on test set
print("\n=== Final Evaluation ===")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Loss
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Get class names
class_names = list(train_generator.class_indices.keys())

# Print classification report
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()


plt.savefig('/scratch/gpfs/ALAINK/Suthi/training_history.png')
plt.savefig('/scratch/gpfs/ALAINK/Suthi/confusion_matrix.png')

# In [ ]:
# Save a few training images with their classifications for the writeup
import matplotlib.pyplot as plt
import os

# Create output folder
OUTPUT_DIR2 = '/scratch/gpfs/ALAINK/Suthi/MidtermReport'
os.makedirs(OUTPUT_DIR2, exist_ok=True)

# Get class names
class_names = list(train_generator.class_indices.keys())

# Get a batch of training images (without augmentation for cleaner images)
sample_gen = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=6,
    class_mode='categorical',
    shuffle=True
)
images, labels = next(sample_gen)

# Get predictions
predictions = model.predict(images)

# Save each image and print predictions
print("=== Predictions ===\n")
for i in range(len(images)):
    true_idx = np.argmax(labels[i])
    pred_idx = np.argmax(predictions[i])
    true_label = class_names[true_idx]
    pred_label = class_names[pred_idx]
    confidence = predictions[i][pred_idx] * 100
    correct = "✓" if true_idx == pred_idx else "✗"
    
    # Print prediction details
    print(f"Image {i+1}: True={true_label}, Predicted={pred_label} ({confidence:.1f}%) {correct}")
    
    # Save image
    filename = f"{OUTPUT_DIR2}/sample_{i+1}_{true_label}.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(images[i])
    plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)")
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

print(f"\n✓ Saved {len(images)} images to {OUTPUT_DIR2}/")

