import tensorflow as tf
import numpy as np
import os


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/brain_tumor_model_finetuned.keras"

TEST_DIR = "dataset/test"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

IMAGES_PER_CLASS = 5


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading fine-tuned model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!\n")


# ============================================================
# TEST EACH CLASS
# ============================================================

total_correct = 0
total_images = 0


for true_class in CLASS_NAMES:

    class_directory = os.path.join(
        TEST_DIR,
        true_class
    )

    if not os.path.exists(class_directory):

        print(
            f"Folder not found: {class_directory}"
        )

        continue

    image_files = [
        file
        for file in os.listdir(class_directory)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    image_files.sort()

    # Take first 5 images
    selected_images = image_files[
        :IMAGES_PER_CLASS
    ]

    correct = 0

    print("\n")
    print("=" * 65)
    print(
        f"TRUE CLASS: {true_class.upper()}"
    )
    print("=" * 65)

    for image_file in selected_images:

        image_path = os.path.join(
            class_directory,
            image_file
        )

        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMG_SIZE
        )

        image_array = tf.keras.utils.img_to_array(
            image
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # IMPORTANT:
        # Do NOT divide by 255.
        # EfficientNet preprocessing is inside the model.

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = np.argmax(
            predictions
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = (
            predictions[predicted_index]
            * 100
        )

        is_correct = (
            predicted_class == true_class
        )

        if is_correct:
            correct += 1
            total_correct += 1
            result = "CORRECT"
        else:
            result = "WRONG"

        total_images += 1

        print(
            f"{image_file:25s} "
            f"-> {predicted_class:12s} "
            f"{confidence:6.2f}% "
            f"[{result}]"
        )

    print(
        f"\nClass Result: "
        f"{correct}/{len(selected_images)} correct"
    )


# ============================================================
# OVERALL RESULT
# ============================================================

print("\n")
print("=" * 65)
print("FINAL 20-IMAGE VALIDATION")
print("=" * 65)

if total_images > 0:

    validation_accuracy = (
        total_correct /
        total_images
    ) * 100

    print(
        f"Correct Predictions: "
        f"{total_correct}/{total_images}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )

print("=" * 65)