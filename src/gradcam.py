import tensorflow as tf
import numpy as np
import cv2
import os

# ==============================
# SETTINGS
# ==============================
MODEL_PATH = "models/brain_tumor_model_finetuned.keras"

IMAGE_PATH = "dataset/test/glioma/Te-gl_1.jpg"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

OUTPUT_DIR = "results/gradcam"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "gradcam_result.png")


# ==============================
# LOAD MODEL
# ==============================
print("Using image:")
print(IMAGE_PATH)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==============================
# FIND EFFICIENTNET MODEL
# ==============================
efficientnet = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        if "efficientnet" in layer.name.lower():
            efficientnet = layer
            break

if efficientnet is None:
    raise ValueError("Could not find EfficientNet model inside the trained model.")

print("\nEfficientNet model:")
print(efficientnet.name)


# ==============================
# FIND LAST CONVOLUTIONAL LAYER
# ==============================
conv_layer = None

for layer in reversed(efficientnet.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        conv_layer = layer
        break

if conv_layer is None:
    raise ValueError("Could not find convolutional layer inside EfficientNet.")

print("Grad-CAM layer:")
print(conv_layer.name)


# ==============================
# LOAD IMAGE
# ==============================
img = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

img_array = tf.keras.utils.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

# EfficientNet preprocessing is included in the model
img_tensor = tf.cast(img_array, tf.float32)


# ==============================
# CREATE GRAD-CAM MODEL
# ==============================
grad_model = tf.keras.models.Model(
    inputs=efficientnet.input,
    outputs=[
        efficientnet.get_layer(conv_layer.name).output,
        efficientnet.output
    ]
)


# ==============================
# GRAD-CAM
# ==============================
with tf.GradientTape() as tape:

    conv_outputs, features = grad_model(img_tensor)

    # Recreate classification head
    x = features

    for layer in model.layers:
        if layer.name == "efficientnetb0":
            continue

        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            x = layer(x)

        elif isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=False)

        elif isinstance(layer, tf.keras.layers.Dense):
            x = layer(x)

    predictions = x

    predicted_class = tf.argmax(predictions[0])

    class_score = predictions[:, predicted_class]


# ==============================
# CALCULATE GRADIENTS
# ==============================
grads = tape.gradient(
    class_score,
    conv_outputs
)

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(heatmap, 0)

heatmap = heatmap / (
    tf.reduce_max(heatmap) + 1e-8
)

heatmap = heatmap.numpy()


# ==============================
# PREDICTION
# ==============================
predicted_index = int(predicted_class)

predicted_label = CLASS_NAMES[predicted_index]

confidence = float(predictions[0][predicted_index]) * 100

print("\nPrediction:")
print("Class:", predicted_label)
print(f"Confidence: {confidence:.2f}%")


# ==============================
# CREATE HEATMAP
# ==============================
original = cv2.imread(IMAGE_PATH)

original = cv2.resize(
    original,
    IMG_SIZE
)

heatmap_uint8 = np.uint8(
    255 * heatmap
)
heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

# Resize Grad-CAM heatmap to match MRI image
heatmap_color = cv2.resize(
    heatmap_color,
    (original.shape[1], original.shape[0])
)

overlay = cv2.addWeighted(
    original,
    0.6,
    heatmap_color,
    0.4,
    0
)


# ==============================
# SAVE RESULT
# ==============================
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

cv2.imwrite(
    OUTPUT_PATH,
    overlay
)

print("\nGrad-CAM result saved:")
print(OUTPUT_PATH)

print("\nGrad-CAM completed successfully!")