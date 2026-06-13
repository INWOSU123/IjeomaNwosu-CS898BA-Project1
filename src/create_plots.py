import matplotlib.pyplot as plt
import os

from config import EDGE_DIR, PLOT_DIR

inputs = sorted([
    f for f in os.listdir(EDGE_DIR)
    if f.endswith("_input.png")
])

for file in inputs:

    root = file.replace("_input.png","")

    images = [
        f"{EDGE_DIR}/{root}_input.png",
        f"{EDGE_DIR}/{root}_sobel.png",
        f"{EDGE_DIR}/{root}_laplacian.png",
        f"{EDGE_DIR}/{root}_canny.png",
        f"{EDGE_DIR}/{root}_prewitt.png"
    ]

    titles = [
        "Input",
        "Sobel",
        "Laplacian",
        "Canny",
        "Prewitt"
    ]

    fig,axs = plt.subplots(
        1,
        5,
        figsize=(20,5)
    )

    for i,path in enumerate(images):

        img = plt.imread(path)

        axs[i].imshow(img,cmap="gray")
        axs[i].set_title(titles[i])
        axs[i].axis("off")

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/{root}_comparison.png"
    )
    #display the images
    plt.show()

    plt.close()