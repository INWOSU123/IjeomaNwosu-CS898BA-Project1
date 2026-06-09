import cv2
import os
import random
import numpy as np

from config import BLURRED_DIR, EDGE_DIR

all_images = [
    os.path.join(BLURRED_DIR,f)
    for f in os.listdir(BLURRED_DIR)
]

random.shuffle(all_images)

subset = all_images[:42]

for img_path in subset:

    img = cv2.imread(img_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    base = os.path.splitext(
        os.path.basename(img_path)
    )[0]

    cv2.imwrite(
        f"{EDGE_DIR}/{base}_input.png",
        img
    )

    # Sobel
    sx = cv2.Sobel(gray, cv2.CV_64F,1,0)
    sy = cv2.Sobel(gray, cv2.CV_64F,0,1)

    sobel = cv2.magnitude(sx,sy)

    cv2.imwrite(
        f"{EDGE_DIR}/{base}_sobel.png",
        sobel
    )

    # Laplacian
    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    cv2.imwrite(
        f"{EDGE_DIR}/{base}_laplacian.png",
        lap
    )

    # Canny
    canny = cv2.Canny(
        gray,
        100,
        200
    )

    cv2.imwrite(
        f"{EDGE_DIR}/{base}_canny.png",
        canny
    )

    # Prewitt

    kernelx = np.array([
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
    ])

    kernely = np.array([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1]
    ])

    px = cv2.filter2D(
        gray,
        -1,
        kernelx
    )

    py = cv2.filter2D(
        gray,
        -1,
        kernely
    )

    prewitt = px + py

    cv2.imwrite(
        f"{EDGE_DIR}/{base}_prewitt.png",
        prewitt
    )