import cv2
import os
import numpy as np

from config import CONVERTED_DIR, TRANSFORMED_DIR

images = [
    "original.png",
    "grayscale.png",
    "binary.png",
    "hsv.png",
    "lab.png",
    "hls.png",
    "hsv_equalized_rgb.png"
]

transformations = [
    ("rot15", ("rotate", 15)),
    ("trans50", ("translate", (50,25))),
    ("rot30", ("rotate", 30)),
    ("scale12", ("scale", 1.2)),
    ("shear10", ("shear", 0.10)),
    ("trans75", ("translate", (75,50))),
    ("rot120", ("rotate", 120)),
    ("scale08", ("scale", 0.8)),
    ("shear20", ("shear", 0.20)),
    ("rot200", ("rotate", 200)),
    ("trans100", ("translate", (100,40))),
    ("scale15", ("scale", 1.5)),
    ("rot270", ("rotate", 270)),
    ("shear30", ("shear", 0.30))
]

idx = 0

for img_name in images:

    img = cv2.imread(
        os.path.join(CONVERTED_DIR,img_name)
    )

    h,w = img.shape[:2]

    for _ in range(2):

        tag,(kind,val) = transformations[idx]
        idx += 1

        if kind == "rotate":

            M = cv2.getRotationMatrix2D(
                (w/2,h/2),
                val,
                1.0
            )

        elif kind == "translate":

            tx,ty = val

            M = np.float32([
                [1,0,tx],
                [0,1,ty]
            ])

        elif kind == "scale":

            M = cv2.getRotationMatrix2D(
                (w/2,h/2),
                0,
                val
            )

        else:

            shear = val

            M = np.float32([
                [1,shear,0],
                [0,1,0]
            ])

        transformed = cv2.warpAffine(
            img,
            M,
            (w,h)
        )

        cv2.imwrite(
            f"{TRANSFORMED_DIR}/{img_name[:-4]}_{tag}.png",
            transformed
        )