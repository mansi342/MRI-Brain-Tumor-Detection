import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

TEST_DIR = "dataset/test"
MODEL_PATH = "models/brain_tumor_resnet50.keras"

# -----------------------------
# LOAD TEST DATASET
# -----------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:", class_names)

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading ResNet50 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# -----------------------------
# MODEL EVALUATION
# -----------------------------
loss, accuracy = model.evaluate(test_ds, verbose=1)

print("\nResNet50 Test Results")
print("----------------------")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# PREDICTIONS
# -----------------------------
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\nClassification Report")
print("----------------------")
print(report)

# Save report
with open(
    "results/resnet50_classification_report.txt",
    "w"
) as f:

    f.write("ResNet50 Classification Report\n")
    f.write("===============================\n\n")
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(report)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("ResNet50 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/resnet50_confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nEvaluation completed successfully!")
print("Report saved:")
print("results/resnet50_classification_report.txt")
print("results/resnet50_confusion_matrix.png")