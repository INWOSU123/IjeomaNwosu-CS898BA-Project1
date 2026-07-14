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
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)


# 1. PROJECT PATHS


HW3_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = HW3_DIR / "dataset" / "test"
MODELS_DIR = HW3_DIR / "models"
RESULTS_DIR = HW3_DIR / "results"

MODEL_PATH = MODELS_DIR / "baseline_best.keras"
CLASS_NAMES_PATH = RESULTS_DIR / "class_names.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. EVALUATION SETTINGS

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32

# 3. CHECK REQUIRED FILES

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset folder was not found: {TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Baseline model was not found: {MODEL_PATH}\n"
        "Run train_baseline.py before evaluating the model."
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file was not found: {CLASS_NAMES_PATH}"
    )

# 4. LOAD CLASS NAMES


with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

number_of_classes = len(class_names)

print("\nClass names:")

for class_index, class_name in enumerate(class_names):
    print(f"{class_index}: {class_name}")


# 5. LOAD TEST DATASET
#
# shuffle=False is important because predictions must remain
# in the same order as the true labels.

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# Improve input performance
test_dataset = test_dataset.prefetch(
    buffer_size=tf.data.AUTOTUNE
)


# 6. LOAD THE BEST BASELINE MODEL

print(f"\nLoading model from: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# 7. EVALUATE TEST LOSS AND ACCURACY

print("\nEvaluating baseline model on test data...")

test_loss, keras_test_accuracy = model.evaluate(
    test_dataset,
    verbose=1,
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy from Keras: {keras_test_accuracy:.4f}")


# 8. COLLECT TRUE LABELS

true_labels = []

for _, labels in test_dataset:
    true_labels.extend(labels.numpy())

true_labels = np.array(true_labels)


# 9. GENERATE PREDICTIONS

print("\nGenerating predictions...")

prediction_probabilities = model.predict(
    test_dataset,
    verbose=1,
)

predicted_labels = np.argmax(
    prediction_probabilities,
    axis=1,
)


# Check that counts match
if len(true_labels) != len(predicted_labels):
    raise ValueError(
        "The number of true labels does not match the number "
        "of model predictions."
    )


# 10. CALCULATE OVERALL METRICS

test_accuracy = accuracy_score(
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


print("\n" + "=" * 60)
print("BASELINE TEST METRICS")
print("=" * 60)

print(f"Accuracy:           {test_accuracy:.4f}")
print(f"Macro precision:    {macro_precision:.4f}")
print(f"Macro recall:       {macro_recall:.4f}")
print(f"Macro F1-score:     {macro_f1:.4f}")
print(f"Weighted precision: {weighted_precision:.4f}")
print(f"Weighted recall:    {weighted_recall:.4f}")
print(f"Weighted F1-score:  {weighted_f1:.4f}")


# 11. CLASSIFICATION REPOORT

report_text = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(number_of_classes)),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print("\nClassification Report:\n")
print(report_text)


# Save text report
report_text_path = (
    RESULTS_DIR / "baseline_classification_report.txt"
)

with open(report_text_path, "w", encoding="utf-8") as file:
    file.write("Baseline Fish CNN Classification Report\n")
    file.write("=" * 45 + "\n\n")
    file.write(f"Test loss: {test_loss:.4f}\n")
    file.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
    file.write(report_text)


# Create report dictionary for CSV
report_dictionary = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(number_of_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_dataframe = pd.DataFrame(
    report_dictionary
).transpose()

report_csv_path = (
    RESULTS_DIR / "baseline_classification_report.csv"
)

report_dataframe.to_csv(
    report_csv_path,
    index=True,
)


# 12. SAVE OVERALL METRICS

overall_metrics = {
    "test_loss": test_loss,
    "accuracy": test_accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1_score": macro_f1,
    "weighted_precision": weighted_precision,
    "weighted_recall": weighted_recall,
    "weighted_f1_score": weighted_f1,
}

metrics_path = RESULTS_DIR / "baseline_test_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(
        overall_metrics,
        file,
        indent=4,
    )


# 13. CONFUSION MATRIX

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(number_of_classes)),
)

print("\nConfusion Matrix:\n")
print(matrix)


# Save matrix as CSV
matrix_dataframe = pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names,
)

matrix_csv_path = (
    RESULTS_DIR / "baseline_confusion_matrix.csv"
)

matrix_dataframe.to_csv(matrix_csv_path)


# Plot confusion matrix
figure, axis = plt.subplots(
    figsize=(10, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)

display.plot(
    ax=axis,
    cmap="Blues",
    values_format="d",
    colorbar=True,
)

axis.set_title(
    "Baseline CNN Confusion Matrix"
)

plt.xticks(
    rotation=45,
    ha="right",
)

plt.tight_layout()

confusion_matrix_path = (
    RESULTS_DIR / "baseline_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# 14. SAMPLE TEST PREDICTIONS

sample_images = []
sample_true_labels = []

for images, labels in test_dataset.take(1):
    sample_images = images.numpy()
    sample_true_labels = labels.numpy()

sample_probabilities = model.predict(
    sample_images,
    verbose=0,
)

sample_predictions = np.argmax(
    sample_probabilities,
    axis=1,
)


number_to_display = min(
    12,
    len(sample_images),
)

plt.figure(figsize=(12, 9))

for position in range(number_to_display):

    true_class = class_names[
        int(sample_true_labels[position])
    ]

    predicted_class = class_names[
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
        f"Actual: {true_class}\n"
        f"Predicted: {predicted_class}\n"
        f"Confidence: {confidence:.2%}",
        fontsize=9,
    )

    plt.axis("off")

plt.tight_layout()

sample_predictions_path = (
    RESULTS_DIR / "baseline_sample_predictions.png"
)

plt.savefig(
    sample_predictions_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# 15. FINISHED

print("\nBaseline evaluation completed successfully.")

print(f"\nSaved classification report:")
print(report_text_path)

print("\nSaved classification report CSV:")
print(report_csv_path)

print("\nSaved overall metrics:")
print(metrics_path)

print("\nSaved confusion matrix:")
print(confusion_matrix_path)

print("\nSaved sample predictions:")
print(sample_predictions_path)