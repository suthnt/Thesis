"""Grad-CAM Visualization for AlexNet Binary Classifier - 5 YEARS DATA
Loads the trained 5-year model and generates heatmaps.
Run AlexNet_5Years.py first to train the model!
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# === CONFIGURATION - 5 YEARS ===
MODEL_PATH = "/scratch/gpfs/ALAINK/Suthi/AlexNet_bin_5Years.keras"
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary_5Years"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/gradcam_outputs_alexnet_5Years"
IMG_SIZE = 227
NUM_SAMPLES = 20
START_INDEX = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output folder (5 Years): {OUTPUT_DIR}")

# === LOAD TRAINED MODEL ===
print(f"Loading 5-Year model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

dummy_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
_ = model(dummy_input)
print("Model graph built.")

last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break
print(f"Last conv layer: {last_conv_layer_name}")

# === GRAD-CAM MODEL ===
last_conv_layer = model.get_layer(last_conv_layer_name)
grad_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = grad_input
for layer in model.layers:
    x = layer(x)
    if layer.name == last_conv_layer_name:
        conv_output = x
        break
pred_output = x
for layer in model.layers[model.layers.index(last_conv_layer)+1:]:
    pred_output = layer(pred_output)
grad_model = tf.keras.Model(inputs=grad_input, outputs=[conv_output, pred_output])

def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_gradcam_visualization(img, heatmap, alpha=0.4):
    import matplotlib.cm as cm
    from PIL import Image
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap_resized)
    heatmap_img = heatmap_img.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    superimposed = jet_heatmap * alpha + img * (1 - alpha)
    return np.clip(superimposed, 0, 1)

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

# === GENERATE GRAD-CAM ===
print(f"\nGenerating Grad-CAM for {NUM_SAMPLES} test images (5 Years)...")
for i in range(NUM_SAMPLES):
    img_array, label = next(test_generator)
    true_class = class_names[np.argmax(label[0])]
    preds = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(preds[0])]
    confidence = np.max(preds[0]) * 100
    heatmap = make_gradcam_heatmap(img_array, grad_model)
    img = img_array[0]
    superimposed = create_gradcam_visualization(img, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title(f"Original\nTrue: {true_class}", fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis('off')
    axes[2].imshow(superimposed)
    color = 'green' if true_class == pred_class else 'red'
    axes[2].set_title(f"Overlay\nPred: {pred_class} ({confidence:.1f}%)", fontsize=12, color=color)
    axes[2].axis('off')
    plt.tight_layout()
    img_index = START_INDEX + i
    correct_str = "correct" if true_class == pred_class else "wrong"
    filename = f"gradcam_5Years_{img_index:03d}_{true_class}_{correct_str}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{i+1}/{NUM_SAMPLES}] {filename}")

print(f"\n✅ Done! Saved to: {OUTPUT_DIR}")

# === SUMMARY GRID ===
print("\nCreating summary grid (5 Years)...")
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
plt.suptitle("Grad-CAM Summary - AlexNet Binary 5 Years", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gradcam_summary_grid_5Years.png"), dpi=200, bbox_inches='tight')
print(f"✅ Saved: gradcam_summary_grid_5Years.png")
