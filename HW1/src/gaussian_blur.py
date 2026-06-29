import cv2

from config import BLURRED_DIR, CONVERTED_DIR, TRANSFORMED_DIR


def main():
    sigmas = [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
    ]

    all_images = []

    for folder in [CONVERTED_DIR, TRANSFORMED_DIR]:
        all_images.extend(sorted(folder.glob("*.png")))

    for image_path in all_images:
        img = cv2.imread(str(image_path))

        if img is None:
            raise FileNotFoundError(f"Image could not be loaded: {image_path}")

        for sigma in sigmas:
            blurred = cv2.GaussianBlur(
                img,
                (0, 0),
                sigmaX=sigma,
            )

            cv2.imwrite(
                str(BLURRED_DIR / f"{image_path.stem}_sigma_{sigma}.png"),
                blurred,
            )


if __name__ == "__main__":
    main()
