import os
import uuid
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

# --------------------------------------------------
# Flask Configuration
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "brain_tumor_model_finetuned.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "app",
    "static",
    "uploads"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "predictions.db"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------------------------------------
# Load Model
# --------------------------------------------------

print("Loading brain tumor model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# --------------------------------------------------
# Class Names
# --------------------------------------------------

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]

IMG_SIZE = (224, 224)


# --------------------------------------------------
# Database Functions
# --------------------------------------------------

def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():

    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# Create database table
init_database()


# --------------------------------------------------
# Grad-CAM Function
# --------------------------------------------------

def make_gradcam(img_array):

    efficientnet_model = None

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            if "efficientnet" in layer.name.lower():

                efficientnet_model = layer
                break

    if efficientnet_model is None:
        raise ValueError("EfficientNet model not found.")

    last_conv_layer = None

    for layer in efficientnet_model.layers:

        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer

    if last_conv_layer is None:
        raise ValueError("Convolutional layer not found.")

    grad_model = tf.keras.models.Model(
        inputs=efficientnet_model.input,
        outputs=[
            last_conv_layer.output,
            efficientnet_model.output
        ]
    )

    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:

        conv_outputs, efficientnet_output = grad_model(
            img_tensor
        )

        x = efficientnet_output

        for layer in model.layers:

            if layer == efficientnet_model:
                continue

            if isinstance(
                layer,
                (
                    tf.keras.layers.GlobalAveragePooling2D,
                    tf.keras.layers.Dropout,
                    tf.keras.layers.Dense
                )
            ):

                x = layer(x)

        predictions = x

        predicted_index = tf.argmax(
            predictions[0]
        )

        class_output = predictions[
            0,
            predicted_index
        ]

    gradients = tape.gradient(
        class_output,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        max_value + 1e-8
    )

    heatmap = heatmap.numpy()

    # Resize heatmap to original image size
    original_height = img_array.shape[1]
    original_width = img_array.shape[2]

    heatmap = cv2.resize(
        heatmap,
        (
            original_width,
            original_height
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original_image = np.uint8(
        img_array[0] * 255
    )

    original_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    )

    superimposed = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return superimposed


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# Prediction Route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":

        return "No image selected."

    allowed_extensions = {
        "jpg",
        "jpeg",
        "png"
    }

    extension = file.filename.rsplit(
        ".",
        1
    )[-1].lower()

    if extension not in allowed_extensions:

        return "Invalid image format."

    # Generate unique filename
    unique_filename = (
        str(uuid.uuid4())
        + "."
        + extension
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    file.save(filepath)

    # --------------------------------------------------
    # Read Image
    # --------------------------------------------------

    image = cv2.imread(filepath)

    if image is None:

        return "Unable to read image."

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_image = image_rgb.copy()

    resized_image = cv2.resize(
        image_rgb,
        IMG_SIZE
    )

    img_array = resized_image.astype(
        np.float32
    ) 

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[0][predicted_index]
        * 100
    )

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    gradcam_image = make_gradcam(
        img_array
    )

    gradcam_filename = (
        "gradcam_"
        + unique_filename
    )

    gradcam_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        gradcam_filename
    )

    cv2.imwrite(
        gradcam_path,
        gradcam_image
    )

    # --------------------------------------------------
    # Save Prediction to Database
    # --------------------------------------------------

    connection = get_db_connection()

    connection.execute(
        """
        INSERT INTO predictions
        (image_name, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            file.filename,
            predicted_class,
            confidence,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    connection.commit()
    connection.close()

    # --------------------------------------------------
    # Show Result
    # --------------------------------------------------

    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=round(
            confidence,
            2
        ),
        image_file=unique_filename,
        gradcam_file=gradcam_filename
    )


# --------------------------------------------------
# Prediction History
# --------------------------------------------------

@app.route("/history")
def history():

    connection = get_db_connection()

    predictions = connection.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return render_template(
        "history.html",
        predictions=predictions
    )
# --------------------------------------------------
# About Page
# --------------------------------------------------

@app.route("/about")
def about():

    return render_template("about.html")


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )