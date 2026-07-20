from pathlib import Path
import csv
import json

import cv2
import numpy as np


# ==========================================================
# Project paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ADVANCED_DIR = PROJECT_ROOT / "Advanced-Segmentation"

RESULTS_IMAGE_DIR = ADVANCED_DIR / "images" / "results"
GROUND_TRUTH_PATH = (
    ADVANCED_DIR
    / "images"
    / "ground_truth"
    / "reference_mask.png"
)

METRICS_OUTPUT_DIR = ADVANCED_DIR / "results"

CLASS_MAPS = {
    "Channel A: Original RGB":
        RESULTS_IMAGE_DIR / "channel_a_class_map.npy",

    "Channel B: HSV V Normalized":
        RESULTS_IMAGE_DIR / "channel_b_class_map.npy",

    "Channel C: RGB Channels Normalized":
        RESULTS_IMAGE_DIR / "channel_c_class_map.npy",
}

# SegFormer predicted the unidentified figure as person.
TARGET_CLASS_ID = 12


# ==========================================================
# Image and mask utilities
# ==========================================================

def load_ground_truth(path: Path) -> np.ndarray:
    """
    Load the Homework Two ground-truth mask and convert it
    into a binary mask containing values 0 and 1.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth mask was not found:\n{path}"
        )

    ground_truth = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if ground_truth is None:
        raise ValueError(
            f"OpenCV could not read the ground-truth mask:\n{path}"
        )

    # Any nonzero pixel becomes foreground.
    binary_ground_truth = (
        ground_truth > 0
    ).astype(np.uint8)

    return binary_ground_truth


def load_class_map(path: Path) -> np.ndarray:
    """
    Load a SegFormer class map saved as a NumPy array.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"SegFormer class map was not found:\n{path}"
        )

    class_map = np.load(path)

    if class_map.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional class map, "
            f"but received shape {class_map.shape}"
        )

    return class_map


def create_binary_prediction(
    class_map: np.ndarray,
    target_class_id: int,
) -> np.ndarray:
    """
    Convert the multi-class SegFormer prediction into a
    binary mask.

    1 = unidentified figure/person
    0 = background
    """

    binary_mask = (
        class_map == target_class_id
    ).astype(np.uint8)

    return binary_mask


def resize_mask_if_needed(
    mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """
    Resize a binary mask when its dimensions do not match
    the target dimensions.

    Nearest-neighbor interpolation preserves class labels.
    """

    target_height, target_width = target_shape

    if mask.shape == target_shape:
        return mask

    resized_mask = cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return (resized_mask > 0).astype(np.uint8)


# ==========================================================
# Evaluation metrics
# ==========================================================

def calculate_iou(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Calculate Intersection over Union.
    """

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth,
    ).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def calculate_dice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Calculate the Dice coefficient.
    """

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    total_pixels = prediction.sum() + ground_truth.sum()

    if total_pixels == 0:
        return 1.0

    return float(
        (2.0 * intersection) / total_pixels
    )


def calculate_pixel_statistics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """
    Calculate pixel-level confusion statistics.
    """

    true_positive = int(
        np.logical_and(
            prediction == 1,
            ground_truth == 1,
        ).sum()
    )

    false_positive = int(
        np.logical_and(
            prediction == 1,
            ground_truth == 0,
        ).sum()
    )

    false_negative = int(
        np.logical_and(
            prediction == 0,
            ground_truth == 1,
        ).sum()
    )

    true_negative = int(
        np.logical_and(
            prediction == 0,
            ground_truth == 0,
        ).sum()
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


# ==========================================================
# Save outputs
# ==========================================================

def save_binary_mask(
    path: Path,
    binary_mask: np.ndarray,
) -> None:
    """
    Save a binary mask using 0 for background and
    255 for foreground.
    """

    display_mask = binary_mask * 255

    success = cv2.imwrite(
        str(path),
        display_mask,
    )

    if not success:
        raise OSError(
            f"Could not save binary mask:\n{path}"
        )


def save_metrics_csv(
    results: list[dict],
    path: Path,
) -> None:
    """
    Save evaluation metrics as a CSV file.
    """

    fieldnames = [
        "channel",
        "target_class_id",
        "iou",
        "dice",
        "predicted_foreground_pixels",
        "ground_truth_foreground_pixels",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def save_metrics_json(
    results: list[dict],
    path: Path,
) -> None:
    """
    Save evaluation metrics as a JSON file.
    """

    with path.open(
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            results,
            json_file,
            indent=4,
        )


# ==========================================================
# Main program
# ==========================================================

def main() -> None:
    METRICS_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Advanced Segmentation Evaluation")
    print("=" * 72)
    print(f"Ground truth: {GROUND_TRUTH_PATH}")
    print(f"Target SegFormer class ID: {TARGET_CLASS_ID}")

    ground_truth = load_ground_truth(
        GROUND_TRUTH_PATH
    )

    print(
        f"Ground-truth dimensions: "
        f"{ground_truth.shape[1]} × {ground_truth.shape[0]}"
    )

    print(
        f"Ground-truth foreground pixels: "
        f"{int(ground_truth.sum()):,}"
    )

    results = []

    for channel_name, class_map_path in CLASS_MAPS.items():

        print("\n" + "-" * 72)
        print(f"Evaluating {channel_name}")
        print(f"Class map: {class_map_path.name}")

        class_map = load_class_map(
            class_map_path
        )

        prediction = create_binary_prediction(
            class_map=class_map,
            target_class_id=TARGET_CLASS_ID,
        )

        ground_truth_for_evaluation = resize_mask_if_needed(
            ground_truth,
            prediction.shape,
        )

        if ground_truth.shape != prediction.shape:
            print(
                "Ground-truth mask resized using "
                "nearest-neighbor interpolation."
            )

        iou = calculate_iou(
            prediction,
            ground_truth_for_evaluation,
        )

        dice = calculate_dice(
            prediction,
            ground_truth_for_evaluation,
        )

        statistics = calculate_pixel_statistics(
            prediction,
            ground_truth_for_evaluation,
        )

        channel_file_name = (
            channel_name
            .lower()
            .replace(":", "")
            .replace(" ", "_")
        )

        binary_mask_path = (
            RESULTS_IMAGE_DIR
            / f"{channel_file_name}_binary_mask.png"
        )

        save_binary_mask(
            binary_mask_path,
            prediction,
        )

        result = {
            "channel": channel_name,
            "target_class_id": TARGET_CLASS_ID,
            "iou": round(iou, 6),
            "dice": round(dice, 6),
            "predicted_foreground_pixels": int(
                prediction.sum()
            ),
            "ground_truth_foreground_pixels": int(
                ground_truth_for_evaluation.sum()
            ),
            **statistics,
        }

        results.append(result)

        print(
            f"Predicted foreground pixels: "
            f"{result['predicted_foreground_pixels']:,}"
        )

        print(f"IoU:  {iou:.4f}")
        print(f"Dice: {dice:.4f}")

    csv_path = (
        METRICS_OUTPUT_DIR
        / "advanced_segmentation_metrics.csv"
    )

    json_path = (
        METRICS_OUTPUT_DIR
        / "advanced_segmentation_metrics.json"
    )

    save_metrics_csv(
        results,
        csv_path,
    )

    save_metrics_json(
        results,
        json_path,
    )

    print("\n" + "=" * 72)
    print("Evaluation completed successfully.")
    print(f"CSV results:  {csv_path}")
    print(f"JSON results: {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()