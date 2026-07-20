from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# ==========================================================
# Project paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADVANCED_DIR = PROJECT_ROOT / "Advanced-Segmentation"

PREPROCESSED_DIR = ADVANCED_DIR / "images" / "preprocessed"
RESULTS_DIR = ADVANCED_DIR / "images" / "results"
GROUND_TRUTH_DIR = ADVANCED_DIR / "images" / "ground_truth"

OUTPUT_PATH = (
    RESULTS_DIR
    / "advanced_segmentation_comparison.png"
)


# ==========================================================
# Image paths
# ==========================================================

IMAGE_PATHS = {
    "Original RGB": (
        PREPROCESSED_DIR
        / "channel_a_original_rgb.png"
    ),

    "HSV V-Channel Normalized": (
        PREPROCESSED_DIR
        / "channel_b_hsv_v_normalized.png"
    ),

    "RGB Channels Normalized": (
        PREPROCESSED_DIR
        / "channel_c_rgb_normalized.png"
    ),

    "Channel A Segmentation": (
        RESULTS_DIR
        / "channel_a_overlay.png"
    ),

    "Channel B Segmentation": (
        RESULTS_DIR
        / "channel_b_overlay.png"
    ),

    "Channel C Segmentation": (
        RESULTS_DIR
        / "channel_c_overlay.png"
    ),

    "Ground Truth": (
        GROUND_TRUTH_DIR
        / "reference_mask.png"
    ),
}


# ==========================================================
# Helper functions
# ==========================================================

def load_color_image(path: Path):
    """
    Load a color image and convert it from BGR to RGB.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required image was not found:\n{path}"
        )

    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise ValueError(
            f"OpenCV could not read:\n{path}"
        )

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )


def load_ground_truth(path: Path):
    """
    Load the ground-truth mask in grayscale.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth mask was not found:\n{path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not read:\n{path}"
        )

    return image


# ==========================================================
# Main program
# ==========================================================

def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Creating Advanced Segmentation Comparison Plot")
    print("=" * 72)

    original = load_color_image(
        IMAGE_PATHS["Original RGB"]
    )

    hsv_normalized = load_color_image(
        IMAGE_PATHS["HSV V-Channel Normalized"]
    )

    rgb_normalized = load_color_image(
        IMAGE_PATHS["RGB Channels Normalized"]
    )

    channel_a = load_color_image(
        IMAGE_PATHS["Channel A Segmentation"]
    )

    channel_b = load_color_image(
        IMAGE_PATHS["Channel B Segmentation"]
    )

    channel_c = load_color_image(
        IMAGE_PATHS["Channel C Segmentation"]
    )

    ground_truth = load_ground_truth(
        IMAGE_PATHS["Ground Truth"]
    )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(20, 10),
    )

    axes = axes.flatten()

    axes[0].imshow(original)
    axes[0].set_title("Original RGB")

    axes[1].imshow(hsv_normalized)
    axes[1].set_title(
        "HSV V-Channel Normalized"
    )

    axes[2].imshow(rgb_normalized)
    axes[2].set_title(
        "RGB Channels Normalized"
    )

    axes[3].imshow(
        ground_truth,
        cmap="gray",
    )
    axes[3].set_title("Ground Truth")

    axes[4].imshow(channel_a)
    axes[4].set_title(
        "Channel A Segmentation"
    )

    axes[5].imshow(channel_b)
    axes[5].set_title(
        "Channel B Segmentation"
    )

    axes[6].imshow(channel_c)
    axes[6].set_title(
        "Channel C Segmentation"
    )

    # Hide the unused eighth panel.
    axes[7].axis("off")

    for axis in axes[:7]:
        axis.axis("off")

    figure.suptitle(
        "Advanced Segmentation Comparison Using SegFormer",
        fontsize=18,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Comparison plot created successfully:"
    )
    print(OUTPUT_PATH)
    print("=" * 72)


if __name__ == "__main__":
    main()