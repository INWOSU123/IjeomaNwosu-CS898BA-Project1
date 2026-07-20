from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


# ==========================================================
# Project paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_DIR = (
    PROJECT_ROOT
    / "Advanced-Segmentation"
    / "images"
    / "preprocessed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Advanced-Segmentation"
    / "images"
    / "results"
)

INPUT_IMAGES = {
    "channel_a": PREPROCESSED_DIR / "channel_a_original_rgb.png",
    "channel_b": PREPROCESSED_DIR / "channel_b_hsv_v_normalized.png",
    "channel_c": PREPROCESSED_DIR / "channel_c_rgb_normalized.png",
}

# SegFormer model trained for semantic segmentation.
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


# ==========================================================
# Visualization helpers
# ==========================================================

def create_color_palette(number_of_classes: int) -> np.ndarray:
    """
    Create a deterministic RGB color for each predicted class.
    """

    random_generator = np.random.default_rng(seed=42)

    palette = random_generator.integers(
        low=0,
        high=256,
        size=(number_of_classes, 3),
        dtype=np.uint8,
    )

    # Keep class 0 dark for clearer background visualization.
    palette[0] = [0, 0, 0]

    return palette


def colorize_segmentation(
    segmentation_map: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """
    Convert integer class IDs into a multi-colored RGB image.
    """

    return palette[segmentation_map]


def create_overlay(
    image_rgb: np.ndarray,
    colored_mask_rgb: np.ndarray,
    alpha: float = 0.50,
) -> np.ndarray:
    """
    Blend the colored segmentation result with the input image.
    """

    overlay = cv2.addWeighted(
        image_rgb,
        1.0 - alpha,
        colored_mask_rgb,
        alpha,
        0,
    )

    return overlay


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    """
    Save an RGB NumPy image using OpenCV.
    """

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(path), image_bgr)

    if not success:
        raise OSError(f"Could not save image:\n{path}")


# ==========================================================
# SegFormer inference
# ==========================================================

def run_inference(
    image_path: Path,
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run SegFormer inference and return:
    1. Original RGB image
    2. Predicted class map
    3. Colored segmentation mask
    """

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image was not found:\n{image_path}"
        )

    pil_image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_image)

    inputs = processor(
        images=pil_image,
        return_tensors="pt",
    )

    inputs = {
        name: value.to(device)
        for name, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    resized_logits = functional.interpolate(
        logits,
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )

    segmentation_map = (
        resized_logits.argmax(dim=1)[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    number_of_classes = model.config.num_labels
    palette = create_color_palette(number_of_classes)

    colored_mask_rgb = colorize_segmentation(
        segmentation_map,
        palette,
    )

    return image_rgb, segmentation_map, colored_mask_rgb


# ==========================================================
# Main program
# ==========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("SegFormer Advanced Segmentation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print("Loading model and image processor...")

    processor = SegformerImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    for channel_name, image_path in INPUT_IMAGES.items():
        print(f"\nProcessing {channel_name}: {image_path.name}")

        (
            image_rgb,
            segmentation_map,
            colored_mask_rgb,
        ) = run_inference(
            image_path=image_path,
            processor=processor,
            model=model,
            device=device,
        )

        overlay_rgb = create_overlay(
            image_rgb=image_rgb,
            colored_mask_rgb=colored_mask_rgb,
        )

        # Save the raw class-ID map.
        np.save(
            OUTPUT_DIR / f"{channel_name}_class_map.npy",
            segmentation_map,
        )

        # Save a grayscale representation of class IDs.
        cv2.imwrite(
            str(OUTPUT_DIR / f"{channel_name}_class_map.png"),
            segmentation_map,
        )

        save_rgb_image(
            OUTPUT_DIR / f"{channel_name}_colored_mask.png",
            colored_mask_rgb,
        )

        save_rgb_image(
            OUTPUT_DIR / f"{channel_name}_overlay.png",
            overlay_rgb,
        )

        unique_classes, pixel_counts = np.unique(
            segmentation_map,
            return_counts=True,
        )

        print("Predicted classes:")

        for class_id, pixel_count in zip(
            unique_classes,
            pixel_counts,
        ):
            label = model.config.id2label.get(
                int(class_id),
                f"class_{class_id}",
            )

            print(
                f"  ID {class_id}: {label} "
                f"({pixel_count:,} pixels)"
            )

    print("\n" + "=" * 70)
    print("SegFormer inference completed successfully.")
    print(f"Results saved to:\n{OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()