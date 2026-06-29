import cv2

from config import MASK_DIR, NORMALIZED_IMAGE, SEGMENT_DIR


def main():
    img = cv2.imread(str(NORMALIZED_IMAGE))

    if img is None:
        raise FileNotFoundError(f"Image could not be loaded: {NORMALIZED_IMAGE}")

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY,
    )

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_fg = cv2.bitwise_and(img, img, mask=otsu)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    adaptive_fg = cv2.bitwise_and(img, img, mask=adaptive)

    cv2.imwrite(str(MASK_DIR / "otsu_mask.png"), otsu)
    cv2.imwrite(str(MASK_DIR / "adaptive_mask.png"), adaptive)
    cv2.imwrite(str(SEGMENT_DIR / "otsu_segment.png"), otsu_fg)
    cv2.imwrite(str(SEGMENT_DIR / "adaptive_segment.png"), adaptive_fg)
    print("Threshold segmentation complete")


if __name__ == "__main__":
    main()
