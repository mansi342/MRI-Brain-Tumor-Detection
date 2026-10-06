import os
import uuid
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# MODEL PATH
# ============================================================

# ResNet50 model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "brain_tumor_resnet50.keras"
)


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "app",
    "static",
    "uploads"
)


# ============================================================
# DATABASE PATH
# ============================================================

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "predictions.db"
)


# Create required folders
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    os.path.dirname(DATABASE_PATH),
    exist_ok=True
)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# LOAD MODEL
# ============================================================

print("========================================")
print("Loading ResNet50 brain tumor model...")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]


# Image size expected by model
IMG_SIZE = (224, 224)


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_db_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()

    connection.close()


# Initialize database
init_database()


# ============================================================
# RESNET50 GRAD-CAM
# ============================================================

def make_gradcam(img_array):

    """
    Generates Grad-CAM visualization
    for the ResNet50 model.
    """

    # --------------------------------------------------------
    # Get ResNet50 base model
    # --------------------------------------------------------

    base_model = model.get_layer(
        "resnet50"
    )


    # --------------------------------------------------------
    # Get last convolutional layer
    # --------------------------------------------------------

    last_conv_layer = base_model.get_layer(
        "conv5_block3_3_conv"
    )


    # --------------------------------------------------------
    # Create Grad-CAM model
    # --------------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )


    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, base_output = grad_model(
            img_array
        )


        # Pass output through classification head
        x = base_output


        # Global Average Pooling
        x = model.get_layer(
            "global_average_pooling2d"
        )(x)


        # Dropout
        x = model.get_layer(
            "dropout"
        )(x)


        # Dense classification layer
        predictions = model.get_layer(
            "dense"
        )(x)


        # Find predicted class
        predicted_index = tf.argmax(
            predictions[0]
        )


        # Score for predicted class
        class_score = predictions[
            0,
            predicted_index
        ]


    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )


    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )


    # Remove batch dimension
    conv_outputs = conv_outputs[0]


    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )


    # Keep only positive values
    heatmap = tf.maximum(
        heatmap,
        0
    )


    # Normalize heatmap
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + 1e-8
    )


    heatmap = heatmap.numpy()


    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    original_height = img_array.shape[1]

    original_width = img_array.shape[2]

    heatmap = cv2.resize(
        heatmap,
        (
            original_width,
            original_height
        )
    )


    # --------------------------------------------------------
    # Convert heatmap to 0-255
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * heatmap
    )


    # --------------------------------------------------------
    # Apply color map
    # --------------------------------------------------------

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    # --------------------------------------------------------
    # Convert original image
    # --------------------------------------------------------

    original_image = np.uint8(
        img_array[0]
    )


    original_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    superimposed = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )


    return superimposed


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # Check uploaded image
    # --------------------------------------------------------

    if "image" not in request.files:

        return "No image uploaded."


    file = request.files["image"]


    if file.filename == "":

        return "No image selected."


    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Generate unique filename
    # --------------------------------------------------------

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


    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        filepath
    )


    if image is None:

        return "Unable to read image."


    # BGR → RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # Keep original image
    original_image = image_rgb.copy()


    # ========================================================
    # RESIZE IMAGE
    # ========================================================

    resized_image = cv2.resize(
        image_rgb,
        IMG_SIZE
    )


    # Convert to float32
    img_array = resized_image.astype(
        np.float32
    )


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(
        img_array,
        verbose=0
    )


    # Get predicted class
    predicted_index = np.argmax(
        predictions[0]
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    # Get confidence
    confidence = float(
        predictions[0][predicted_index]
        * 100
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    try:

        gradcam_image = make_gradcam(
            img_array
        )

    except Exception as e:

        print(
            "Grad-CAM error:",
            str(e)
        )

        return (
            "Prediction successful, "
            "but Grad-CAM generation failed."
        )


    # --------------------------------------------------------
    # Save Grad-CAM
    # --------------------------------------------------------

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


    # ========================================================
    # SAVE PREDICTION TO DATABASE
    # ========================================================

    connection = get_db_connection()


    connection.execute(
        """
        INSERT INTO predictions
        (
            image_name,
            prediction,
            confidence,
            created_at
        )
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


    # ========================================================
    # SHOW RESULT
    # ========================================================

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


# ============================================================
# PREDICTION HISTORY
# ============================================================

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


# ============================================================
# ABOUT PAGE
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )