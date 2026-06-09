import cv2
import os

from config import (
    CONVERTED_DIR,
    TRANSFORMED_DIR,
    BLURRED_DIR
)

sigmas = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5
]

all_images = []

for folder in [CONVERTED_DIR, TRANSFORMED_DIR]:

    for f in os.listdir(folder):

        if f.endswith(".png"):

            all_images.append(
                os.path.join(folder,f)
            )

for image_path in all_images:

    img = cv2.imread(image_path)

    base = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    for sigma in sigmas:

        blurred = cv2.GaussianBlur(
            img,
            (0,0),
            sigmaX=sigma
        )

        cv2.imwrite(
            f"{BLURRED_DIR}/{base}_sigma_{sigma}.png",
            blurred
        )