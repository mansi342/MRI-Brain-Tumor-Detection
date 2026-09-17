import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "models/brain_tumor_model_finetuned.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

TEST_IMAGES = [
    "dataset/test/glioma/Te-gl_1.jpg",
    "dataset/test/meningioma/Te-me_1.jpg",
    "dataset/test/no_tumor/Te-no_1.jpg",
    "dataset/test/pituitary/Te-pi_1.jpg"
]

print("Loading fine-tuned model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!\n")


for image_path in TEST_IMAGES:

    print("=" * 60)
    print("Image:", image_path)

    if not os.path.exists(image_path):
        print("FILE NOT FOUND")
        continue

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)

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

    predicted_index = np.argmax(predictions)

    print(
        "Predicted:",
        CLASS_NAMES[predicted_index]
    )

    print(
        f"Confidence: "
        f"{predictions[predicted_index] * 100:.2f}%"
    )

    print("\nAll probabilities:")

    for i, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name:12s}: "
            f"{predictions[i] * 100:.2f}%"
        )