import matplotlib.pyplot as plt

from config import COMPARISON_DIR, IMAGE_PATH, MASK_DIR, NORMALIZED_IMAGE


def main():
    files = [
        IMAGE_PATH,
        NORMALIZED_IMAGE,
        MASK_DIR / "otsu_mask.png",
        MASK_DIR / "adaptive_mask.png",
        MASK_DIR / "kmeans_mask.png",
    ]
    titles = ["Original", "Normalized", "Otsu", "Adaptive", "KMeans"]

    fig, ax = plt.subplots(
        1,
        5,
        figsize=(22, 5),
    )

    for i, path in enumerate(files):
        print("Loading:", path)
        img = plt.imread(path)
        print("Shape:", img.shape)
        ax[i].imshow(img, cmap="gray")
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "final_plot.png")
    plt.show()
    #plt.close(fig)


if __name__ == "__main__":
    main()
