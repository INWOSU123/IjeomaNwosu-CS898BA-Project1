from pathlib import Path
import gc
import itertools
import json
import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


# 1. PROJECT PATHS

HW3_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = HW3_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"

MODELS_DIR = HW3_DIR / "models"
RESULTS_DIR = HW3_DIR / "results"
TUNING_DIR = RESULTS_DIR / "tuning"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TUNING_DIR.mkdir(parents=True, exist_ok=True)


# 2. GENERAL SETTINGS

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)


# 3. HYPERPARAMETERS TO TEST

LEARNING_RATES = [
    0.01, 0.001, 0.0001
]

BATCH_SIZES = [
    32, 64
]

DROPOUT_RATES = [
    0.30, 0.50
]


# 4. VERIFY REQUIRED FOLDERS

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory was not found: {TRAIN_DIR}"
    )

if not VALID_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory was not found: {VALID_DIR}"
    )


# 5. GET CLASS NAMES

class_names = sorted(
    [
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    ]
)

number_of_classes = len(class_names)

if number_of_classes < 2:
    raise ValueError(
        "At least two fish class folders are required."
    )

print("\nFish classes:")

for class_index, class_name in enumerate(class_names):
    print(f"{class_index}: {class_name}")

print(f"\nNumber of classes: {number_of_classes}")


# 6. DATASET CREATION FUNCTION

def create_datasets(batch_size):
    """
    Load the training and validation datasets using the
    selected batch size.
    """

    train_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=batch_size,
            shuffle=True,
            seed=RANDOM_SEED,
        )
    )

    validation_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            VALID_DIR,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=batch_size,
            shuffle=False,
        )
    )

    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return train_dataset, validation_dataset

# 7. MODEL CREATION FUNCTION

def build_model(learning_rate, dropout_rate):
    """
    Build a new CNN from random initialization.
    A completely new model is created for each experiment.
    No pretrained model or weights are used.
    """

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

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                    3,
                )
            ),

            data_augmentation,

            tf.keras.layers.Rescaling(
                1.0 / 255.0
            ),

            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                units=128,
                activation="relu",
            ),

            tf.keras.layers.Dropout(
                rate=dropout_rate
            ),

            tf.keras.layers.Dense(
                units=number_of_classes,
                activation="softmax",
            ),
        ],
        name="tuned_fish_cnn",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# 8. CREATE ALL EXPERIMENT COMBINATIONS

experiments = list(
    itertools.product(
        LEARNING_RATES,
        BATCH_SIZES,
        DROPOUT_RATES,
    )
)

print(
    f"\nTotal tuning experiments: {len(experiments)}"
)


# 9. RUN GRID SEARCH

experiment_results = []

best_validation_loss = float("inf")
best_experiment = None

overall_start_time = time.time()


for experiment_number, (
    learning_rate,
    batch_size,
    dropout_rate,
) in enumerate(experiments, start=1):

    print("\n" + "=" * 70)

    print(
        f"EXPERIMENT {experiment_number} "
        f"OF {len(experiments)}"
    )

    print("=" * 70)

    print(f"Learning rate: {learning_rate}")
    print(f"Batch size:    {batch_size}")
    print(f"Dropout rate:  {dropout_rate}")

    experiment_name = (
        f"experiment_{experiment_number:02d}"
        f"_lr_{learning_rate}"
        f"_batch_{batch_size}"
        f"_dropout_{dropout_rate}"
    )

    experiment_folder = (
        TUNING_DIR / experiment_name
    )

    experiment_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clear the previous TensorFlow model from memory.
    tf.keras.backend.clear_session()
    gc.collect()

    # Reset the seed 
    tf.random.set_seed(RANDOM_SEED)

    train_dataset, validation_dataset = (
        create_datasets(batch_size)
    )

    model = build_model(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
    )

    temporary_model_path = (
        experiment_folder / "best_model.keras"
    )

    checkpoint_callback = (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=temporary_model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        )
    )

    early_stopping_callback = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    )

    csv_logger_callback = (
        tf.keras.callbacks.CSVLogger(
            filename=(
                experiment_folder
                / "training_history.csv"
            )
        )
    )

    experiment_start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            csv_logger_callback,
        ],
        verbose=1,
    )

    experiment_runtime_seconds = (
        time.time() - experiment_start_time
    )

    completed_epochs = len(
        history.history["loss"]
    )

    minimum_validation_loss = min(
        history.history["val_loss"]
    )

    best_validation_accuracy = max(
        history.history["val_accuracy"]
    )

    best_epoch_index = (
        history.history["val_loss"].index(
            minimum_validation_loss
        )
    )

    best_epoch_number = best_epoch_index + 1

    experiment_record = {
        "experiment_number": experiment_number,
        "experiment_name": experiment_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch_number,
        "minimum_validation_loss": (
            minimum_validation_loss
        ),
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "runtime_seconds": (
            experiment_runtime_seconds
        ),
    }

    experiment_results.append(
        experiment_record
    )

    print("\nExperiment result:")

    print(
        f"Minimum validation loss: "
        f"{minimum_validation_loss:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Best epoch: {best_epoch_number}"
    )

    print(
        f"Runtime: "
        f"{experiment_runtime_seconds:.1f} seconds"
    )

    # Save this experiment's history plot.
    epoch_numbers = range(
        1,
        completed_epochs + 1,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        epoch_numbers,
        history.history["loss"],
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epoch_numbers,
        history.history["val_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"Experiment {experiment_number}: "
        f"Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        experiment_folder / "loss_curve.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    # Check whether this is the best experiment so far.
    if minimum_validation_loss < best_validation_loss:

        best_validation_loss = (
            minimum_validation_loss
        )

        best_experiment = (
            experiment_record.copy()
        )

        # Save a separate copy as the overall best model.
        best_model_path = MODELS_DIR / "optimized_best.keras"

        best_saved_model = tf.keras.models.load_model(
            temporary_model_path
        )

        best_saved_model.save(best_model_path)

        print(
            "\nNew overall best model saved to:"
        )

        print(best_model_path)


# 10. SAVE ALL EXPERIMENT RESULTS

results_dataframe = pd.DataFrame(
    experiment_results
)

results_dataframe = results_dataframe.sort_values(
    by=[
        "minimum_validation_loss",
        "best_validation_accuracy",
    ],
    ascending=[
        True,
        False,
    ],
)

results_csv_path = (
    RESULTS_DIR / "hyperparameter_results.csv"
)

results_dataframe.to_csv(
    results_csv_path,
    index=False,
)


# Save as JSON as well.
results_json_path = (
    RESULTS_DIR / "hyperparameter_results.json"
)

with open(
    results_json_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        experiment_results,
        file,
        indent=4,
    )


# 11. SAVE THE BEST CONFIGURATION

best_config_path = (
    RESULTS_DIR / "best_hyperparameters.json"
)

with open(
    best_config_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        best_experiment,
        file,
        indent=4,
    )


# 12. CREATE COMPARISON PLOT

plot_dataframe = results_dataframe.copy()

plot_dataframe["configuration"] = (
    "LR="
    + plot_dataframe[
        "learning_rate"
    ].astype(str)
    + "\nB="
    + plot_dataframe[
        "batch_size"
    ].astype(str)
    + "\nD="
    + plot_dataframe[
        "dropout_rate"
    ].astype(str)
)

plt.figure(figsize=(14, 8))

plt.bar(
    plot_dataframe["configuration"],
    plot_dataframe[
        "minimum_validation_loss"
    ],
)

plt.xlabel("Hyperparameter Configuration")
plt.ylabel("Minimum Validation Loss")

plt.title(
    "Hyperparameter Tuning Results"
)

plt.xticks(
    rotation=45,
    ha="right",
)

plt.tight_layout()

comparison_plot_path = (
    RESULTS_DIR
    / "hyperparameter_validation_loss.png"
)

plt.savefig(
    comparison_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# 13. FINAL SUMMARY

total_runtime_seconds = (
    time.time() - overall_start_time
)

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 70)

print(
    f"\nTotal experiments: "
    f"{len(experiments)}"
)

print(
    f"Total runtime: "
    f"{total_runtime_seconds / 60:.2f} minutes"
)

print("\nBest configuration:")

print(
    f"Learning rate: "
    f"{best_experiment['learning_rate']}"
)

print(
    f"Batch size: "
    f"{best_experiment['batch_size']}"
)

print(
    f"Dropout rate: "
    f"{best_experiment['dropout_rate']}"
)

print(
    f"Minimum validation loss: "
    f"{best_experiment['minimum_validation_loss']:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_experiment['best_validation_accuracy']:.4f}"
)

print(
    f"Best epoch: "
    f"{best_experiment['best_epoch']}"
)

print("\nBest optimized model:")
print(MODELS_DIR / "optimized_best.keras")

print("\nExperiment table:")
print(results_csv_path)

print("\nBest configuration file:")
print(best_config_path)

print("\nComparison plot:")
print(comparison_plot_path)