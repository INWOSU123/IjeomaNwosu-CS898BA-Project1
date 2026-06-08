import os

IMAGE_PATH = "images/original/HW1_IMG_CS898BA.png"

CONVERTED_DIR = "images/converted"
TRANSFORMED_DIR = "images/transformed"
BLURRED_DIR = "images/blurred"
EDGE_DIR = "images/edges"
PLOT_DIR = "images/plots"

for folder in [
    CONVERTED_DIR,
    TRANSFORMED_DIR,
    BLURRED_DIR,
    EDGE_DIR,
    PLOT_DIR
]:
    os.makedirs(folder, exist_ok=True)