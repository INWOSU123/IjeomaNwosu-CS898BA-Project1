from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# 1. PROJECT PATHS

HW3_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = HW3_DIR / "dataset" / "test"
MODELS_DIR = HW3_DIR / "models"
RESULTS_DIR = HW3_DIR / "results"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_best.keras"
OPTIMIZED_MODEL_PATH = MODELS_DIR / "optimized_best.keras"

CLASS_NAMES_PATH = RESULTS_DIR / "class_names.json"
BEST_PARAMETERS_PATH = RESULTS_DIR / "best_hyperparameters.json"

BASELINE_HISTORY_PATH = (
    RESULTS_DIR / "baseline_training_history.csv"
)

FINAL_RESULTS_DIR = RESULTS_DIR / "final_comparison"
FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 2. SETTINGS

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32


# 3. VERIFY REQUIRED FILES

required_paths = [
    TEST_DIR,
    BASELINE_MODEL_PATH,
    OPTIMIZED_MODEL_PATH,
    CLASS_NAMES_PATH,
    BASELINE_HISTORY_PATH,
    BEST_PARAMETERS_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file or folder was not found:\n{path}"
        )


# 4. LOAD CLASS NAMES AND BEST CONFIGURATION

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8",
) as file:
    class_names = json.load(file)

with open(
    BEST_PARAMETERS_PATH,
    "r",
    encoding="utf-8",
) as file:
    best_parameters = json.load(file)

number_of_classes = len(class_names)

print("\nFish classes:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print("\nBest hyperparameters:")

print(
    "Learning rate:",
    best_parameters["learning_rate"],
)

print(
    "Batch size:",
    best_parameters["batch_size"],
)

print(
    "Dropout rate:",
    best_parameters["dropout_rate"],
)


# 5. LOCATE OPTIMIZED TRAINING HISTORY

best_experiment_name = best_parameters["experiment_name"]

optimized_history_path = (
    RESULTS_DIR
    / "tuning"
    / best_experiment_name
    / "training_history.csv"
)

if not optimized_history_path.exists():
    raise FileNotFoundError(
        "The optimized training history was not found:\n"
        f"{optimized_history_path}"
    )


# 6. LOAD THE UNTOUCHED TEST DATASET

test_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
)

test_dataset = test_dataset.prefetch(
    buffer_size=tf.data.AUTOTUNE
)


# 7. COLLECT TRUE TEST LABELS

true_labels = []

for _, labels in test_dataset:
    true_labels.extend(labels.numpy())

true_labels = np.asarray(true_labels)


# 8. MODEL EVALUATION FUNCTION

def evaluate_model(model_path, model_name):
    """
    Load one trained model and evaluate it on the test dataset.
    """

    print("\n" + "=" * 65)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 65)

    model = tf.keras.models.load_model(model_path)

    test_loss, keras_accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    probabilities = model.predict(
        test_dataset,
        verbose=1,
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(number_of_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_dictionary = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(number_of_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(number_of_classes)),
    )

    metrics = {
        "model": model_name,
        "test_loss": float(test_loss),
        "keras_accuracy": float(keras_accuracy),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    print(f"\nTest loss:       {test_loss:.4f}")
    print(f"Test accuracy:   {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")

    print("\nClassification report:\n")
    print(report_text)

    safe_name = model_name.lower().replace(" ", "_")

    # Save text classification report
    report_text_path = (
        FINAL_RESULTS_DIR
        / f"{safe_name}_classification_report.txt"
    )

    with open(
        report_text_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"{model_name} Classification Report\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"Test loss: {test_loss:.4f}\n")
        file.write(f"Test accuracy: {accuracy:.4f}\n\n")
        file.write(report_text)

    # Save report as CSV
    report_dataframe = pd.DataFrame(
        report_dictionary
    ).transpose()

    report_dataframe.to_csv(
        FINAL_RESULTS_DIR
        / f"{safe_name}_classification_report.csv"
    )

    # Save confusion matrix as CSV
    matrix_dataframe = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    matrix_dataframe.to_csv(
        FINAL_RESULTS_DIR
        / f"{safe_name}_confusion_matrix.csv"
    )

    return {
        "model": model,
        "metrics": metrics,
        "predictions": predicted_labels,
        "probabilities": probabilities,
        "confusion_matrix": matrix,
    }


# 9. EVALUATE BOTH MODELS

baseline_result = evaluate_model(
    BASELINE_MODEL_PATH,
    "Baseline Model",
)

optimized_result = evaluate_model(
    OPTIMIZED_MODEL_PATH,
    "Optimized Model",
)


# 10. SAVE METRIC COMPARISON

comparison_dataframe = pd.DataFrame(
    [
        baseline_result["metrics"],
        optimized_result["metrics"],
    ]
)

comparison_csv_path = (
    FINAL_RESULTS_DIR / "model_metric_comparison.csv"
)

comparison_dataframe.to_csv(
    comparison_csv_path,
    index=False,
)

comparison_json_path = (
    FINAL_RESULTS_DIR / "model_metric_comparison.json"
)

with open(
    comparison_json_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        comparison_dataframe.to_dict(
            orient="records"
        ),
        file,
        indent=4,
    )


# 11. CALCULATE IMPROVEMENT

baseline_accuracy = baseline_result[
    "metrics"
]["accuracy"]

optimized_accuracy = optimized_result[
    "metrics"
]["accuracy"]

baseline_f1 = baseline_result[
    "metrics"
]["macro_f1"]

optimized_f1 = optimized_result[
    "metrics"
]["macro_f1"]

accuracy_change = (
    optimized_accuracy - baseline_accuracy
)

f1_change = optimized_f1 - baseline_f1

print("\n" + "=" * 65)
print("BASELINE VS. OPTIMIZED COMPARISON")
print("=" * 65)

print(
    f"Baseline accuracy:  {baseline_accuracy:.4f}"
)

print(
    f"Optimized accuracy: {optimized_accuracy:.4f}"
)

print(
    f"Accuracy change:    {accuracy_change:+.4f}"
)

print(
    f"\nBaseline macro F1:  {baseline_f1:.4f}"
)

print(
    f"Optimized macro F1: {optimized_f1:.4f}"
)

print(
    f"Macro F1 change:    {f1_change:+.4f}"
)


# 12. LOAD TRAINING HISTORIES


baseline_history = pd.read_csv(
    BASELINE_HISTORY_PATH
)

optimized_history = pd.read_csv(
    optimized_history_path
)

# 13. CREATE REQUIRED FINAL IMAGE GRID
#
# Grid includes:
# 1. Baseline accuracy
# 2. Optimized accuracy
# 3. Baseline loss
# 4. Optimized loss
# 5. Optimized confusion matrix
# 6. Test metric comparison


figure, axes = plt.subplots(
    2,
    3,
    figsize=(20, 12),
)


# Baseline accuracy

axes[0, 0].plot(
    baseline_history["epoch"] + 1,
    baseline_history["accuracy"],
    label="Training",
)

axes[0, 0].plot(
    baseline_history["epoch"] + 1,
    baseline_history["val_accuracy"],
    label="Validation",
)

axes[0, 0].set_title(
    "Baseline Accuracy"
)

axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(True)


# Optimized accuracy

axes[0, 1].plot(
    optimized_history["epoch"] + 1,
    optimized_history["accuracy"],
    label="Training",
)

axes[0, 1].plot(
    optimized_history["epoch"] + 1,
    optimized_history["val_accuracy"],
    label="Validation",
)

axes[0, 1].set_title(
    "Optimized Accuracy"
)

axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True)


# Test metrics

metric_names = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
]

baseline_metric_values = [
    baseline_result["metrics"]["accuracy"],
    baseline_result["metrics"]["macro_precision"],
    baseline_result["metrics"]["macro_recall"],
    baseline_result["metrics"]["macro_f1"],
]

optimized_metric_values = [
    optimized_result["metrics"]["accuracy"],
    optimized_result["metrics"]["macro_precision"],
    optimized_result["metrics"]["macro_recall"],
    optimized_result["metrics"]["macro_f1"],
]

positions = np.arange(
    len(metric_names)
)

bar_width = 0.35

axes[0, 2].bar(
    positions - bar_width / 2,
    baseline_metric_values,
    bar_width,
    label="Baseline",
)

axes[0, 2].bar(
    positions + bar_width / 2,
    optimized_metric_values,
    bar_width,
    label="Optimized",
)

axes[0, 2].set_title(
    "Test Metric Comparison"
)

axes[0, 2].set_ylabel("Score")
axes[0, 2].set_ylim(0, 1)

axes[0, 2].set_xticks(
    positions
)

axes[0, 2].set_xticklabels(
    metric_names,
    rotation=25,
    ha="right",
)

axes[0, 2].legend()
axes[0, 2].grid(axis="y")


# Baseline loss

axes[1, 0].plot(
    baseline_history["epoch"] + 1,
    baseline_history["loss"],
    label="Training",
)

axes[1, 0].plot(
    baseline_history["epoch"] + 1,
    baseline_history["val_loss"],
    label="Validation",
)

axes[1, 0].set_title(
    "Baseline Loss"
)

axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Optimized loss

axes[1, 1].plot(
    optimized_history["epoch"] + 1,
    optimized_history["loss"],
    label="Training",
)

axes[1, 1].plot(
    optimized_history["epoch"] + 1,
    optimized_history["val_loss"],
    label="Validation",
)

axes[1, 1].set_title(
    "Optimized Loss"
)

axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].legend()
axes[1, 1].grid(True)


# Optimized confusion matrix

optimized_matrix = optimized_result[
    "confusion_matrix"
]

confusion_image = axes[1, 2].imshow(
    optimized_matrix,
    interpolation="nearest",
    cmap="Blues",
)

axes[1, 2].set_title(
    "Optimized Model Confusion Matrix"
)

axes[1, 2].set_xlabel(
    "Predicted Class"
)

axes[1, 2].set_ylabel(
    "Actual Class"
)

axes[1, 2].set_xticks(
    np.arange(number_of_classes)
)

axes[1, 2].set_yticks(
    np.arange(number_of_classes)
)

axes[1, 2].set_xticklabels(
    class_names,
    rotation=45,
    ha="right",
)

axes[1, 2].set_yticklabels(
    class_names
)

threshold = optimized_matrix.max() / 2

for row in range(number_of_classes):
    for column in range(number_of_classes):

        cell_value = optimized_matrix[
            row,
            column,
        ]

        axes[1, 2].text(
            column,
            row,
            str(cell_value),
            ha="center",
            va="center",
            color=(
                "white"
                if cell_value > threshold
                else "black"
            ),
        )

figure.colorbar(
    confusion_image,
    ax=axes[1, 2],
    fraction=0.046,
    pad=0.04,
)

figure.suptitle(
    "Fish Classification: Baseline and Optimized CNN Comparison",
    fontsize=18,
)

plt.tight_layout(
    rect=(0, 0, 1, 0.96)
)

final_grid_path = (
    FINAL_RESULTS_DIR
    / "final_model_comparison_grid.png"
)

plt.savefig(
    final_grid_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# 14. CREATE OPTIMIZED SAMPLE PREDICTIONS

sample_images = []
sample_labels = []

for images, labels in test_dataset.take(1):
    sample_images = images.numpy()
    sample_labels = labels.numpy()

sample_probabilities = optimized_result[
    "model"
].predict(
    sample_images,
    verbose=0,
)

sample_predictions = np.argmax(
    sample_probabilities,
    axis=1,
)

number_to_show = min(
    12,
    len(sample_images),
)

plt.figure(figsize=(12, 9))

for position in range(number_to_show):

    actual_name = class_names[
        int(sample_labels[position])
    ]

    predicted_name = class_names[
        int(sample_predictions[position])
    ]

    confidence = float(
        np.max(sample_probabilities[position])
    )

    plt.subplot(3, 4, position + 1)

    plt.imshow(
        sample_images[position].astype("uint8")
    )

    plt.title(
        f"Actual: {actual_name}\n"
        f"Predicted: {predicted_name}\n"
        f"Confidence: {confidence:.1%}",
        fontsize=9,
    )

    plt.axis("off")

plt.tight_layout()

sample_path = (
    FINAL_RESULTS_DIR
    / "optimized_sample_predictions.png"
)

plt.savefig(
    sample_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# 15. FINISHED

print("\nFinal comparison completed successfully.")

print("\nMetric comparison:")
print(comparison_csv_path)

print("\nFinal visualization grid:")
print(final_grid_path)

print("\nOptimized sample predictions:")
print(sample_path)