import random

import cv2
import numpy as np

from config import BLURRED_DIR, EDGE_DIR


def main():
    all_images = sorted(BLURRED_DIR.glob("*.png"))

    random.seed(898)
    random.shuffle(all_images)

    subset = all_images[:42]

    for img_path in subset:
        img = cv2.imread(str(img_path))

        if img is None:
            raise FileNotFoundError(f"Image could not be loaded: {img_path}")

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY,
        )

        base = img_path.stem

        cv2.imwrite(
            str(EDGE_DIR / f"{base}_input.png"),
            img,
        )

        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobel = cv2.magnitude(sx, sy)
        sobel = cv2.convertScaleAbs(sobel)
        cv2.imwrite(
            str(EDGE_DIR / f"{base}_sobel.png"),
            sobel,
        )

        lap = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        )
        lap = cv2.convertScaleAbs(lap)
        cv2.imwrite(
            str(EDGE_DIR / f"{base}_laplacian.png"),
            lap,
        )

        canny = cv2.Canny(
            gray,
            100,
            200,
        )
        cv2.imwrite(
            str(EDGE_DIR / f"{base}_canny.png"),
            canny,
        )

        kernelx = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ])

        kernely = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ])

        px = cv2.filter2D(
            gray,
            cv2.CV_64F,
            kernelx,
        )

        py = cv2.filter2D(
            gray,
            cv2.CV_64F,
            kernely,
        )

        prewitt = cv2.magnitude(px, py)
        prewitt = cv2.convertScaleAbs(prewitt)
        cv2.imwrite(
            str(EDGE_DIR / f"{base}_prewitt.png"),
            prewitt,
        )


if __name__ == "__main__":
    main()
