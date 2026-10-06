import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

# 8 GB RAM + CPU training ke liye safer
BATCH_SIZE = 8

EPOCHS = 15

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


# ============================================================
# 3. LOAD VALIDATION DATA
# ============================================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 4. GET CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(i, "->", class_name)

print("\nNumber of classes:", len(class_names))


# ============================================================
# 5. PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="data_augmentation")


# ============================================================
# 7. LOAD PRETRAINED RESNET50
# ============================================================

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False


# ============================================================
# 8. BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    4,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 10. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 11. CALLBACKS
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "brain_tumor_resnet50.keras"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    model_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nStarting ResNet50 training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]
)


# ============================================================
# 13. SAVE FINAL MODEL
# ============================================================

model.save(model_path)

print("\nResNet50 model saved successfully!")
print("Location:", model_path)


# ============================================================
# 14. PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("ResNet50 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

accuracy_path = os.path.join(
    RESULTS_DIR,
    "resnet50_accuracy.png"
)

plt.savefig(accuracy_path)
plt.close()


# ============================================================
# 15. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ResNet50 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

loss_path = os.path.join(
    RESULTS_DIR,
    "resnet50_loss.png"
)

plt.savefig(loss_path)
plt.close()


print("\nTraining graphs saved:")
print(accuracy_path)
print(loss_path)

print("\nResNet50 training completed successfully!")