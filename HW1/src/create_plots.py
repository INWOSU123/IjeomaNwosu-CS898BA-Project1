import matplotlib.pyplot as plt

from config import EDGE_DIR, PLOT_DIR


def main():
    inputs = sorted(EDGE_DIR.glob("*_input.png"))

    for file in inputs:
        root = file.name.replace("_input.png", "")

        images = [
            EDGE_DIR / f"{root}_input.png",
            EDGE_DIR / f"{root}_sobel.png",
            EDGE_DIR / f"{root}_laplacian.png",
            EDGE_DIR / f"{root}_canny.png",
            EDGE_DIR / f"{root}_prewitt.png",
        ]

        titles = [
            "Input",
            "Sobel",
            "Laplacian",
            "Canny",
            "Prewitt",
        ]

        fig, axs = plt.subplots(
            1,
            5,
            figsize=(20, 5),
        )

        for i, path in enumerate(images):
            img = plt.imread(path)
            axs[i].imshow(img, cmap="gray")
            axs[i].set_title(titles[i])
            axs[i].axis("off")

        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{root}_comparison.png")
        #plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
