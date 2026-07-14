from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


# Project paths

HW3_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = HW3_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

RESULTS_DIR = HW3_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Dataset settings

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32
RANDOM_SEED = 42


# Confirm the folders exist

for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")


# Load the training dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED,
)

# Load the validation dataset


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# Load the test dataset

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# Save class names before dataset transformations
class_names = train_dataset.class_names
number_of_classes = len(class_names)

print("\nFish classes:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

print(f"\nNumber of classes: {number_of_classes}")


# Data augmentation

# These transformations run only during training.

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            mode="horizontal",
            seed=RANDOM_SEED,
        ),
        tf.keras.layers.RandomRotation(
            factor=0.08,
            fill_mode="reflect",
            seed=RANDOM_SEED,
        ),
        tf.keras.layers.RandomBrightness(
            factor=0.15,
            value_range=(0, 255),
            seed=RANDOM_SEED,
        ),
    ],
    name="fish_data_augmentation",
)


# Normalization

# Converts pixels from 0–255 to 0–1.

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)

# Improve dataset performance

autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=autotune)
validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
test_dataset = test_dataset.prefetch(buffer_size=autotune)

# Display one batch of original training images

for images, labels in train_dataset.take(1):

    plt.figure(figsize=(12, 8))

    for position in range(min(12, len(images))):

        plt.subplot(3, 4, position + 1)

        image = images[position].numpy().astype("uint8")
        class_index = tf.argmax(labels[position]).numpy()

        plt.imshow(image)
        plt.title(class_names[class_index])
        plt.axis("off")

    plt.tight_layout()

    original_output = RESULTS_DIR / "training_samples.png"
    plt.savefig(original_output, dpi=200, bbox_inches="tight")
    #plt.show()
    plt.close()

    print(f"\nSaved original samples to: {original_output}")


# Display augmented training images

for images, labels in train_dataset.take(1):

    augmented_images = data_augmentation(images, training=True)

    plt.figure(figsize=(12, 8))

    for position in range(min(12, len(augmented_images))):

        plt.subplot(3, 4, position + 1)

        augmented_image = tf.clip_by_value(
            augmented_images[position],
            0,
            255,
        )

        augmented_image = augmented_image.numpy().astype("uint8")
        class_index = tf.argmax(labels[position]).numpy()

        plt.imshow(augmented_image)
        plt.title(class_names[class_index])
        plt.axis("off")

    plt.tight_layout()

    augmented_output = RESULTS_DIR / "augmented_training_samples.png"
    plt.savefig(augmented_output, dpi=200, bbox_inches="tight")
    #plt.show()
    plt.close()

    print(f"Saved augmented samples to: {augmented_output}")


# Verify normalization

for images, _ in train_dataset.take(1):

    normalized_images = normalization_layer(images)

    print("\nBefore normalization:")
    print("Minimum pixel value:", float(tf.reduce_min(images)))
    print("Maximum pixel value:", float(tf.reduce_max(images)))

    print("\nAfter normalization:")
    print("Minimum pixel value:", float(tf.reduce_min(normalized_images)))
    print("Maximum pixel value:", float(tf.reduce_max(normalized_images)))

    break


print("\nData pipeline completed successfully.")