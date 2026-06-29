import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import affine_transformations
import conversions
import create_plots
import edge_detection
import gaussian_blur
import image_stats


def main():
    image_stats.main()
    conversions.main()
    affine_transformations.main()
    gaussian_blur.main()
    edge_detection.main()
    create_plots.main()


if __name__ == "__main__":
    main()
