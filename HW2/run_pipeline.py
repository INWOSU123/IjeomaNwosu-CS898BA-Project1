import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import create_segmentation_plot
import kmeans_segmentation
import metrics
import normalize
import threshold_segmentation


def main():
    normalize.main()
    threshold_segmentation.main()
    kmeans_segmentation.main()
    metrics.main()
    create_segmentation_plot.main()


if __name__ == "__main__":
    main()
