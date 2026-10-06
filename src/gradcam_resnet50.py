import tensorflow as tf
import numpy as np
import cv2
import os
import glob

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/brain_tumor_resnet50.keras"

OUTPUT_PATH = "results/gradcam/resnet50_gradcam_result.jpg"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


# ============================================================
# FIND TEST IMAGE
# ============================================================

image_files = glob.glob("dataset/test/glioma/*")

if not image_files:
    raise FileNotFoundError(
        "No images found inside dataset/test/glioma/"
    )

IMAGE_PATH = image_files[0]

print("Using image:", IMAGE_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ResNet50 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# GET RESNET50 BASE MODEL
# ============================================================

base_model = model.get_layer("resnet50")

print(
    "Base model:",
    base_model.name
)


# ============================================================
# GET LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = base_model.get_layer(
    "conv5_block3_3_conv"
)

print(
    "Grad-CAM layer:",
    last_conv_layer.name
)


# ============================================================
# LOAD IMAGE
# ============================================================

original = cv2.imread(
    IMAGE_PATH
)

if original is None:

    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# Convert BGR → RGB
original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)


# Resize
resized = cv2.resize(
    original_rgb,
    IMG_SIZE
)


# Convert to float32
img_array = resized.astype(
    np.float32
)


# Add batch dimension
img_array = np.expand_dims(
    img_array,
    axis=0
)


print("Image loaded successfully.")


# ============================================================
# CREATE GRAD-CAM MODEL
# ============================================================

# Model that returns:
# 1. Last convolutional feature maps
# 2. Final prediction

grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)


# ============================================================
# RUN GRAD-CAM
# ============================================================

print("\nCalculating Grad-CAM...")


with tf.GradientTape() as tape:

    # Get convolutional output and base model output
    conv_outputs, base_output = grad_model(
        img_array
    )

    # Pass base output through remaining layers
    x = model.get_layer(
        "global_average_pooling2d"
    )(base_output)

    x = model.get_layer(
        "dropout"
    )(x)

    predictions = model.get_layer(
        "dense"
    )(x)

    # Predicted class
    predicted_class = tf.argmax(
        predictions[0]
    )

    # Score of predicted class
    class_score = predictions[
        :, predicted_class
    ]


# ============================================================
# CALCULATE GRADIENTS
# ============================================================

grads = tape.gradient(
    class_score,
    conv_outputs
)


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)


# Remove batch dimension
conv_outputs = conv_outputs[0]


# ============================================================
# CREATE HEATMAP
# ============================================================

heatmap = conv_outputs @ (
    pooled_grads[..., tf.newaxis]
)

heatmap = tf.squeeze(
    heatmap
)


# ============================================================
# NORMALIZE HEATMAP
# ============================================================

heatmap = tf.maximum(
    heatmap,
    0
)

max_value = tf.reduce_max(
    heatmap
)

if max_value > 0:

    heatmap /= max_value

heatmap = heatmap.numpy()


# ============================================================
# RESIZE HEATMAP
# ============================================================

heatmap = cv2.resize(
    heatmap,
    (
        original_rgb.shape[1],
        original_rgb.shape[0]
    )
)


# Convert to 0–255
heatmap_uint8 = np.uint8(
    255 * heatmap
)


# ============================================================
# APPLY COLOR MAP
# ============================================================

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)


# BGR → RGB
heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# CREATE OVERLAY
# ============================================================

overlay = cv2.addWeighted(
    original_rgb,
    0.6,
    heatmap_color,
    0.4,
    0
)


# ============================================================
# PREDICTION
# ============================================================

predicted_index = int(
    predicted_class.numpy()
)

predicted_label = CLASS_NAMES[
    predicted_index
]

confidence = float(
    predictions[0][predicted_index]
) * 100


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================================")
print("          PREDICTION")
print("================================")

print(
    "Predicted Class:",
    predicted_label
)

print(
    f"Confidence: {confidence:.2f}%"
)


print("\nClass probabilities:")

for i, class_name in enumerate(
    CLASS_NAMES
):

    probability = float(
        predictions[0][i]
    ) * 100

    print(
        f"{class_name}: "
        f"{probability:.2f}%"
    )


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    "results/gradcam",
    exist_ok=True
)

cv2.imwrite(
    OUTPUT_PATH,
    cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )
)


# ============================================================
# FINISHED
# ============================================================

print("\n================================")
print("        GRAD-CAM COMPLETED")
print("================================")

print(
    "Grad-CAM saved successfully!"
)

print(
    "Location:",
    OUTPUT_PATH
)