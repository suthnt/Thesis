"""Grad-CAM Visualization for AlexNet Binary Classifier
Loads a trained model and generates heatmaps showing what the model focuses on.
No retraining needed!
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# === CONFIGURATION ===
MODEL_PATH = "/scratch/gpfs/ALAINK/Suthi/AlexNet_bin.keras"  # Your trained model
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/gradcam_outputs"
IMG_SIZE = 227
NUM_SAMPLES = 20  # Number of test images to visualize
START_INDEX = 20  # Start numbering from this index (to add to existing)

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output folder: {OUTPUT_DIR}")

# === LOAD TRAINED MODEL ===
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Warm-up call to build the model graph (required for Keras 3)
dummy_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
_ = model(dummy_input)
print("Model graph built.")

# Find the last convolutional layer name
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

print(f"Last conv layer: {last_conv_layer_name}")

# === GRAD-CAM FUNCTIONS ===
# Build a separate model for getting conv layer activations (Keras 3 compatible)
last_conv_layer = model.get_layer(last_conv_layer_name)

# Create a new input that matches the model's expected input
grad_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Build feature extractor: input -> last conv layer
x = grad_input
for layer in model.layers:
    x = layer(x)
    if layer.name == last_conv_layer_name:
        conv_output = x
        break

# Build the rest of the model for predictions
pred_output = x
for layer in model.layers[model.layers.index(last_conv_layer)+1:]:
    pred_output = layer(pred_output)

# Create the grad model
grad_model = tf.keras.Model(inputs=grad_input, outputs=[conv_output, pred_output])
print("Grad-CAM model built successfully!")


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    """Generate Grad-CAM heatmap for a given image (Keras 3 compatible)."""
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)  # Explicitly watch conv outputs
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Gradient of predicted class w.r.t. conv layer output
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Average gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv outputs by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize to 0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def create_gradcam_visualization(img, heatmap, alpha=0.4):
    """Overlay heatmap on original image."""
    # Resize heatmap to image size
    heatmap_resized = np.uint8(255 * heatmap)
    
    # Use matplotlib to resize and apply colormap
    import matplotlib.cm as cm
    from PIL import Image
    
    # Resize heatmap
    heatmap_img = Image.fromarray(heatmap_resized)
    heatmap_img = heatmap_img.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img)
    
    # Apply jet colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    
    # Superimpose
    superimposed = jet_heatmap * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)
    
    return superimposed


# === LOAD TEST DATA ===
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=True
)

class_names = list(test_generator.class_indices.keys())
print(f"Classes: {class_names}")

# === GENERATE GRAD-CAM VISUALIZATIONS ===
print(f"\nGenerating Grad-CAM for {NUM_SAMPLES} test images...")

for i in range(NUM_SAMPLES):
    # Get a test image
    img_array, label = next(test_generator)
    true_class = class_names[np.argmax(label[0])]
    
    # Get prediction
    preds = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(preds[0])]
    confidence = np.max(preds[0]) * 100
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, grad_model)
    
    # Create visualization
    img = img_array[0]  # Remove batch dimension
    superimposed = create_gradcam_visualization(img, heatmap)
    
    # Plot original + heatmap + overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f"Original\nTrue: {true_class}", fontsize=12)
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(superimposed)
    color = 'green' if true_class == pred_class else 'red'
    axes[2].set_title(f"Overlay\nPred: {pred_class} ({confidence:.1f}%)", fontsize=12, color=color)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save with offset index
    img_index = START_INDEX + i
    correct_str = "correct" if true_class == pred_class else "wrong"
    filename = f"gradcam_{img_index:03d}_{true_class}_{correct_str}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [{i+1}/{NUM_SAMPLES}] {filename} - True: {true_class}, Pred: {pred_class} ({confidence:.1f}%)")

print(f"\n✅ Done! Saved {NUM_SAMPLES} Grad-CAM visualizations to: {OUTPUT_DIR}")

# === CREATE SUMMARY GRID FOR NEW BATCH ===
print("\nCreating summary grid for new batch...")

# Reset generator and get fresh samples
test_generator.reset()
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for i in range(min(20, NUM_SAMPLES)):
    img_array, label = next(test_generator)
    true_class = class_names[np.argmax(label[0])]
    
    preds = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(preds[0])]
    confidence = np.max(preds[0]) * 100
    
    heatmap = make_gradcam_heatmap(img_array, grad_model)
    superimposed = create_gradcam_visualization(img_array[0], heatmap)
    
    axes[i].imshow(superimposed)
    color = 'green' if true_class == pred_class else 'red'
    axes[i].set_title(f"T:{true_class}\nP:{pred_class} ({confidence:.0f}%)", fontsize=10, color=color)
    axes[i].axis('off')

plt.suptitle(f"Grad-CAM Summary (Batch {START_INDEX//20 + 1}) - AlexNet Binary Classifier", fontsize=16)
plt.tight_layout()
grid_filename = f"gradcam_summary_grid_batch{START_INDEX//20 + 1}.png"
plt.savefig(os.path.join(OUTPUT_DIR, grid_filename), dpi=200, bbox_inches='tight')
print(f"✅ Saved summary grid: {OUTPUT_DIR}/{grid_filename}")
