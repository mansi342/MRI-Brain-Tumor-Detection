
import os
import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "dataset/test"

MODEL_PATH = (
    "models/brain_tumor_model_finetuned.keras"
)

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 3. CLASS NAMES
# ============================================================

class_names = test_dataset.class_names

print("\nClasses:")

for i, class_name in enumerate(class_names):

    print(
        i,
        "->",
        class_name
    )


# ============================================================
# 4. LOAD FINE-TUNED MODEL
# ============================================================

print("\nLoading fine-tuned model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Fine-tuned model loaded successfully!"
)


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

print("\nEvaluating model on test dataset...\n")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\n")
print("=" * 60)

print(
    f"Fine-tuned Test Loss: {test_loss:.4f}"
)

print(
    f"Fine-tuned Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# 6. GET PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)

y_pred = np.array(y_pred)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=class_names,

    digits=4
)

print("\nClassification Report:\n")

print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "finetuned_classification_report.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(
        "Fine-Tuned EfficientNetB0 "
        "Classification Report\n\n"
    )

    file.write(
        report
    )

    file.write(
        f"\n\nTest Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nConfusion Matrix:\n")

print(cm)


# ============================================================
# 9. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(cm)

plt.title(
    "Fine-Tuned EfficientNetB0 "
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)


for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


cm_path = os.path.join(
    RESULTS_DIR,
    "finetuned_confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. FINISHED
# ============================================================

print("\nResults saved:")

print(
    report_path
)

print(
    cm_path
)

print("\n")
print("=" * 60)
print(
    "FINE-TUNED MODEL EVALUATION COMPLETED!"
)
print("=" * 60)