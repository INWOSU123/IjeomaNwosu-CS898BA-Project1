import cv2
import numpy as np

from config import MASK_DIR, NORMALIZED_IMAGE, SEGMENT_DIR


def main():
    img = cv2.imread(str(NORMALIZED_IMAGE))

    if img is None:
        raise FileNotFoundError(f"Image could not be loaded: {NORMALIZED_IMAGE}")

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV,
    )
    pixels = hsv.reshape((-1, 3))
    pixels = np.float32(pixels)

    k = 4
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2,
    )

    cv2.setRNGSeed(898)
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    labels = labels.flatten()

    target = 2
    mask = (labels == target).astype(np.uint8) * 255
    mask = mask.reshape(img.shape[:2])

    segment = cv2.bitwise_and(
        img,
        img,
        mask=mask,
    )

    cv2.imwrite(str(MASK_DIR / "kmeans_mask.png"), mask)
    cv2.imwrite(str(SEGMENT_DIR / "kmeans_segment.png"), segment)
    print("KMeans complete")


if __name__ == "__main__":
    main()
