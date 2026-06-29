import cv2

from config import IMAGE_PATH, NORMALIZED_IMAGE


def main():
    img = cv2.imread(str(IMAGE_PATH))

    if img is None:
        raise FileNotFoundError(f"Image could not be loaded: {IMAGE_PATH}")

    b, g, r = cv2.split(img)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)

    normalized = cv2.merge([b, g, r])
    cv2.imwrite(str(NORMALIZED_IMAGE), normalized)
    print("Normalization complete")


if __name__ == "__main__":
    main()
