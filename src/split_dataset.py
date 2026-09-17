import os
import random
import shutil

SOURCE_DIR = "dataset/training_original"
TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

CLASSES = [
    "glioma",
    "meningioma",
    "pituitary",
    "no_tumor"
]

VALIDATION_RATIO = 0.20

random.seed(42)


def create_folder(path):
    os.makedirs(path, exist_ok=True)


for class_name in CLASSES:

    source_class = os.path.join(SOURCE_DIR, class_name)
    train_class = os.path.join(TRAIN_DIR, class_name)
    validation_class = os.path.join(VALIDATION_DIR, class_name)

    create_folder(train_class)
    create_folder(validation_class)

    images = [
        file for file in os.listdir(source_class)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    random.shuffle(images)

    validation_count = int(len(images) * VALIDATION_RATIO)

    validation_images = images[:validation_count]
    train_images = images[validation_count:]

    print(f"\nClass: {class_name}")
    print(f"Total images: {len(images)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(validation_images)}")

    for image in train_images:
        source = os.path.join(source_class, image)
        destination = os.path.join(train_class, image)
        shutil.copy2(source, destination)

    for image in validation_images:
        source = os.path.join(source_class, image)
        destination = os.path.join(validation_class, image)
        shutil.copy2(source, destination)


print("\nDataset splitting completed successfully!")