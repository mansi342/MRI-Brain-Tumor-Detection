import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "dataset/test"
MODEL_PATH = "models/brain_tumor_model.keras"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(i, "->", class_name)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# 4. EVALUATE MODEL
# ============================================================

print("\nEvaluating model on test dataset...\n")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print("==============================")


# ============================================================
# 5. GET TRUE LABELS AND PREDICTIONS
# ============================================================

true_labels = []
predicted_labels = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted_classes
    )


true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    digits=4
)

print(report)


# Save report
report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as file:
    file.write(report)

print("Classification report saved:")
print(report_path)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 8. PLOT CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title("Brain Tumor Classification Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(cm_path)
plt.close()

print("\nConfusion matrix saved:")
print(cm_path)


print("\nEvaluation completed successfully!")