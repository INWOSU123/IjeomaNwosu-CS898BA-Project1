from pathlib import Path
import json

import matplotlib.pyplot as plt
import tensorflow as tf


# 1. PROJECT PATHS


HW3_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = HW3_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"

MODELS_DIR = HW3_DIR / "models"
RESULTS_DIR = HW3_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 2. TRAINING SETTINGS


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42


# 3. CHECK DATASET FOLDERS


if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder was not found: {TRAIN_DIR}"
    )

if not VALID_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder was not found: {VALID_DIR}"
    )

# 4. LOAD TRAINING DATASET


train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED,
)



# 5. LOAD VALIDATION DATASET


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# Get class names before applying prefetch
class_names = train_dataset.class_names
number_of_classes = len(class_names)

print("\nFish classes found:")

for class_number, class_name in enumerate(class_names):
    print(f"{class_number}: {class_name}")

print(f"\nNumber of classes: {number_of_classes}")



class_names_path = RESULTS_DIR / "class_names.json"

with open(class_names_path, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)



# 6. DATA AUGMENTATION
#
# These transformations are applied only during training.


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
    name="training_augmentation",
)



# 7. IMPROVE DATA LOADING PERFORMANCE


autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=autotune
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=autotune
)



# 8. BUILD CUSTOM CNN FROM SCRATCH
#
# No pretrained model or pretrained weights are used.


model = tf.keras.Sequential(
    [
        # Explicit input dimensions
        tf.keras.layers.Input(
            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        ),

        # Training-only augmentation
        data_augmentation,

        # Normalize pixels from [0, 255] to [0, 1]
        tf.keras.layers.Rescaling(1.0 / 255.0),

    
        # Convolution block 1
        
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        
        # Convolution block 2
        
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # Convolution block 3

        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convert feature maps into one vector
        tf.keras.layers.Flatten(),

        # Fully connected hidden layer
        tf.keras.layers.Dense(
            units=128,
            activation="relu",
        ),

        # Baseline regularization
        tf.keras.layers.Dropout(rate=0.30),

        # One probability for each fish class
        tf.keras.layers.Dense(
            units=number_of_classes,
            activation="softmax",
        ),
    ],
    name="baseline_fish_cnn",
)


# 9. COMPILE MODEL


optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# Display model architecture
model.summary()


# Save the architecture summary as a text file

summary_path = RESULTS_DIR / "baseline_model_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    model.summary(
        print_fn=lambda line: file.write(line + "\n")
    )


# 10. CALLBACKS

best_model_path = MODELS_DIR / "baseline_best.keras"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

csv_logger_callback = tf.keras.callbacks.CSVLogger(
    filename=RESULTS_DIR / "baseline_training_history.csv"
)


# 11. TRAIN MODEL

print("\nStarting baseline training...")
print("No pretrained model or pretrained weights are being used.\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        csv_logger_callback,
    ],
)

# 12. SAVE FINAL MODEL

final_model_path = MODELS_DIR / "baseline_final.keras"

model.save(final_model_path)

print(f"\nBest model saved to: {best_model_path}")
print(f"Final model saved to: {final_model_path}")


# 13. PLOT TRAINING AND VALIDATION CURVES

training_accuracy = history.history["accuracy"]
validation_accuracy = history.history["val_accuracy"]

training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]

completed_epochs = range(
    1,
    len(training_accuracy) + 1
)


# Accuracy plot

plt.figure(figsize=(9, 6))

plt.plot(
    completed_epochs,
    training_accuracy,
    marker="o",
    label="Training Accuracy",
)

plt.plot(
    completed_epochs,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline CNN: Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

accuracy_plot_path = (
    RESULTS_DIR / "baseline_accuracy_curve.png"
)

plt.savefig(
    accuracy_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# Loss plot

plt.figure(figsize=(9, 6))

plt.plot(
    completed_epochs,
    training_loss,
    marker="o",
    label="Training Loss",
)

plt.plot(
    completed_epochs,
    validation_loss,
    marker="o",
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline CNN: Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_plot_path = (
    RESULTS_DIR / "baseline_loss_curve.png"
)

plt.savefig(
    loss_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# 14. PRINT FINAL RESULTS

best_validation_accuracy = max(validation_accuracy)
lowest_validation_loss = min(validation_loss)

print("\nBaseline training completed successfully.")

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.4f}"
)

print(
    f"Lowest validation loss: "
    f"{lowest_validation_loss:.4f}"
)

print(f"Accuracy curve saved to: {accuracy_plot_path}")
print(f"Loss curve saved to: {loss_plot_path}")
print(f"Training history saved to: {RESULTS_DIR}")