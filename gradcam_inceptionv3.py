"""
Grad-CAM Visualization for InceptionV3 Binary Classifier (1-year)
Loads best_InceptionV3_1year.keras and generates heatmaps.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# === CONFIGURATION ===
MODEL_PATH = "/scratch/gpfs/ALAINK/Suthi/best_InceptionV3_1year.keras"
DATASET_DIR = "/scratch/gpfs/ALAINK/Suthi/OrganizedDatasetBalancedBinary"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/gradcam_inceptionv3_outputs"
IMG_SIZE = 299  # InceptionV3 expects 299x299
NUM_SAMPLES = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"=== Grad-CAM for InceptionV3 ===")
print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")

# === LOAD MODEL ===
model = tf.keras.models.load_model(MODEL_PATH)
dummy_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
_ = model(dummy_input)

# Use base_model output - run forward pass in GradientTape (avoids Keras 3 Model graph issues)
base_model = model.layers[0]
print(f"Using base_model output for Grad-CAM (shape: (batch, 8, 8, 2048))")


def make_gradcam_heatmap(img_array, model, base_model, pred_index=None):
    """Compute Grad-CAM by running model in parts inside GradientTape."""
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs = base_model(img_tensor)
        tape.watch(conv_outputs)
        x = conv_outputs
        for layer in model.layers[1:]:
            x = layer(x)
        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_np = conv_outputs[0].numpy()
    heatmap = np.dot(conv_np, pooled_grads.numpy())
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


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
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode="categorical",
    shuffle=True,
)
class_names = list(test_generator.class_indices.keys())

# === GENERATE GRAD-CAM ===
print(f"\nGenerating Grad-CAM for {NUM_SAMPLES} images...")
for i in range(NUM_SAMPLES):
    img_array, label = next(test_generator)
    true_class = class_names[np.argmax(label[0])]
    preds = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(preds[0])]
    confidence = np.max(preds[0]) * 100
    heatmap = make_gradcam_heatmap(img_array, model, base_model)
    superimposed = create_gradcam_visualization(img_array[0], heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array[0])
    axes[0].set_title(f"Original\nTrue: {true_class}")
    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    color = "green" if true_class == pred_class else "red"
    axes[2].imshow(superimposed)
    axes[2].set_title(f"Overlay\nPred: {pred_class} ({confidence:.1f}%)", color=color)
    axes[2].axis("off")
    plt.tight_layout()
    correct_str = "correct" if true_class == pred_class else "wrong"
    plt.savefig(os.path.join(OUTPUT_DIR, f"gradcam_inceptionv3_{i:03d}_{true_class}_{correct_str}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [{i+1}/{NUM_SAMPLES}] True: {true_class}, Pred: {pred_class} ({confidence:.1f}%)")

# Summary grid
test_generator.reset()
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()
for i in range(min(20, NUM_SAMPLES)):
    img_array, label = next(test_generator)
    true_class = class_names[np.argmax(label[0])]
    preds = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(preds[0])]
    confidence = np.max(preds[0]) * 100
    heatmap = make_gradcam_heatmap(img_array, model, base_model)
    superimposed = create_gradcam_visualization(img_array[0], heatmap)
    axes[i].imshow(superimposed)
    color = "green" if true_class == pred_class else "red"
    axes[i].set_title(f"T:{true_class}\nP:{pred_class} ({confidence:.0f}%)", fontsize=10, color=color)
    axes[i].axis("off")
plt.suptitle("Grad-CAM Summary - InceptionV3 Binary Classifier", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gradcam_inceptionv3_summary_grid.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"\nDone! Outputs: {OUTPUT_DIR}")
