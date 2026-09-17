import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

BASELINE_MODEL_PATH = "models/brain_tumor_model.keras"

FINETUNED_MODEL_PATH = (
    "models/brain_tumor_model_finetuned.keras"
)

RESULTS_DIR = "results"

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

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

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 4. PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 5. LOAD EXISTING BASELINE MODEL
# ============================================================

print("\nLoading baseline model...")

model = tf.keras.models.load_model(
    BASELINE_MODEL_PATH
)

print("Baseline model loaded successfully!")


# ============================================================
# 6. FIND EFFICIENTNETB0
# ============================================================

efficientnet_model = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        if "efficientnet" in layer.name.lower():

            efficientnet_model = layer
            break


if efficientnet_model is None:

    raise ValueError(
        "EfficientNet model not found."
    )


print(
    "\nEfficientNet model found:",
    efficientnet_model.name
)

print(
    "Total EfficientNet layers:",
    len(efficientnet_model.layers)
)


# ============================================================
# 7. UNFREEZE LAST 30 LAYERS
# ============================================================

print("\nPreparing fine-tuning...")

efficientnet_model.trainable = True

FINE_TUNE_LAYERS = 30

total_layers = len(
    efficientnet_model.layers
)

freeze_until = max(
    0,
    total_layers - FINE_TUNE_LAYERS
)


for i, layer in enumerate(
    efficientnet_model.layers
):

    if i < freeze_until:

        layer.trainable = False

    else:

        # Keep Batch Normalization layers frozen
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

        else:

            layer.trainable = True


print(
    "\nFine-tuning configuration:"
)

print(
    "Frozen layers:",
    freeze_until
)

print(
    "Fine-tuning last layers:",
    FINE_TUNE_LAYERS
)


# ============================================================
# 8. DISPLAY TRAINABLE LAYERS
# ============================================================

trainable_count = sum(
    1
    for layer in efficientnet_model.layers
    if layer.trainable
)

print(
    "Trainable EfficientNet layers:",
    trainable_count
)


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 10. DISPLAY MODEL
# ============================================================

print("\nModel summary:")

model.summary()


# ============================================================
# 11. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True,

    verbose=1
)


checkpoint = ModelCheckpoint(

    FINETUNED_MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ============================================================
# 12. FINE-TUNE MODEL
# ============================================================

print("\n")
print("=" * 60)
print("STARTING EFFICIENTNETB0 FINE-TUNING")
print("=" * 60)
print("\n")


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================================
# 13. LOAD BEST FINE-TUNED MODEL
# ============================================================

print("\nLoading best fine-tuned model...")

model = tf.keras.models.load_model(
    FINETUNED_MODEL_PATH
)

print(
    "Best fine-tuned model loaded successfully!"
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

model.save(
    FINETUNED_MODEL_PATH
)

print(
    "\nFine-tuned model saved at:"
)

print(
    FINETUNED_MODEL_PATH
)


# ============================================================
# 15. PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Fine-Tuning Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Fine-Tuning Validation Accuracy"
)

plt.title(
    "EfficientNetB0 Fine-Tuning Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

accuracy_path = os.path.join(
    RESULTS_DIR,
    "finetune_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Fine-Tuning Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Fine-Tuning Validation Loss"
)

plt.title(
    "EfficientNetB0 Fine-Tuning Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

loss_path = os.path.join(
    RESULTS_DIR,
    "finetune_loss.png"
)

plt.savefig(
    loss_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. FINISHED
# ============================================================

print("\nFine-tuning graphs saved:")

print(
    accuracy_path
)

print(
    loss_path
)

print("\n")
print("=" * 60)
print("FINE-TUNING COMPLETED SUCCESSFULLY!")
print("=" * 60)