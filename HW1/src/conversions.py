import cv2

from config import CONVERTED_DIR, IMAGE_PATH


def main():
    print("Loading image:", IMAGE_PATH)

    img = cv2.imread(str(IMAGE_PATH))

    if img is None:
        raise FileNotFoundError(f"Image could not be loaded: {IMAGE_PATH}")

    print("Image loaded successfully.")

    cv2.imwrite(str(CONVERTED_DIR / "original.png"), img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(CONVERTED_DIR / "grayscale.png"), gray)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    cv2.imwrite(str(CONVERTED_DIR / "binary.png"), binary)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(str(CONVERTED_DIR / "hsv.png"), hsv)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imwrite(str(CONVERTED_DIR / "lab.png"), lab)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    cv2.imwrite(str(CONVERTED_DIR / "hls.png"), hls)

    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])

    rgb_equalized = cv2.cvtColor(
        hsv_eq,
        cv2.COLOR_HSV2BGR,
    )
    cv2.imwrite(
        str(CONVERTED_DIR / "hsv_equalized_rgb.png"),
        rgb_equalized,
    )


if __name__ == "__main__":
    main()
