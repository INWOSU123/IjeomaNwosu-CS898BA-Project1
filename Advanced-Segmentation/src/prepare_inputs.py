from pathlib import Path

import cv2
import numpy as np


# ==========================================================
# Project paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_IMAGE = PROJECT_ROOT / "HW1" / "images" / "converted" / "original.png"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Advanced-Segmentation"
    / "images"
    / "preprocessed"
)

CHANNEL_A_PATH = OUTPUT_DIR / "channel_a_original_rgb.png"
CHANNEL_B_PATH = OUTPUT_DIR / "channel_b_hsv_v_normalized.png"
CHANNEL_C_PATH = OUTPUT_DIR / "channel_c_rgb_normalized.png"


# ==========================================================
# Helper functions
# ==========================================================

def load_image(path: Path) -> np.ndarray:
    """
    Load an image using OpenCV.

    OpenCV reads images in BGR format.
    """

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load the input image:\n{path}"
        )

    return image


def create_channel_a(image_bgr: np.ndarray) -> np.ndarray:
    """
    Channel A:
    Original, unmodified RGB image.

    The image is returned in BGR format because OpenCV uses BGR
    when saving images.
    """

    return image_bgr.copy()


def create_channel_b(image_bgr: np.ndarray) -> np.ndarray:
    """
    Channel B:
    Convert the image to HSV, normalize only the V channel,
    then convert the result back to BGR.
    """

    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    normalized_v = cv2.normalize(
        v_channel,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )

    normalized_hsv = cv2.merge(
        [h_channel, s_channel, normalized_v]
    )

    normalized_bgr = cv2.cvtColor(
        normalized_hsv,
        cv2.COLOR_HSV2BGR,
    )

    return normalized_bgr


def create_channel_c(image_bgr: np.ndarray) -> np.ndarray:
    """
    Channel C:
    Normalize the blue, green, and red channels independently.
    """

    blue, green, red = cv2.split(image_bgr)

    normalized_blue = cv2.normalize(
        blue,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )

    normalized_green = cv2.normalize(
        green,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )

    normalized_red = cv2.normalize(
        red,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )

    normalized_bgr = cv2.merge(
        [
            normalized_blue,
            normalized_green,
            normalized_red,
        ]
    )

    return normalized_bgr


def save_image(path: Path, image: np.ndarray) -> None:
    """
    Save an image and raise an error if the operation fails.
    """

    success = cv2.imwrite(str(path), image)

    if not success:
        raise OSError(
            f"Could not save image:\n{path}"
        )


# ==========================================================
# Main pipeline
# ==========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Advanced Segmentation Input Preparation")
    print("=" * 70)
    print(f"Input image: {INPUT_IMAGE}")
    print(f"Output folder: {OUTPUT_DIR}")

    original_image = load_image(INPUT_IMAGE)

    channel_a = create_channel_a(original_image)
    channel_b = create_channel_b(original_image)
    channel_c = create_channel_c(original_image)

    save_image(CHANNEL_A_PATH, channel_a)
    save_image(CHANNEL_B_PATH, channel_b)
    save_image(CHANNEL_C_PATH, channel_c)

    print("\nGenerated files:")
    print(f"Channel A: {CHANNEL_A_PATH.name}")
    print(f"Channel B: {CHANNEL_B_PATH.name}")
    print(f"Channel C: {CHANNEL_C_PATH.name}")

    print("\nInput image dimensions:")
    print(f"Width: {original_image.shape[1]}")
    print(f"Height: {original_image.shape[0]}")
    print(f"Channels: {original_image.shape[2]}")

    print("\nInput preparation completed successfully.")


if __name__ == "__main__":
    main()