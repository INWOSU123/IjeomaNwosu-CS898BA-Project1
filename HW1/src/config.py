from pathlib import Path

HW_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HW_DIR.parent

IMAGE_PATH = PROJECT_ROOT / "images" / "original" / "HW1_IMG_CS898BA.png"

CONVERTED_DIR = HW_DIR / "images" / "converted"
TRANSFORMED_DIR = HW_DIR / "images" / "transformed"
BLURRED_DIR = HW_DIR / "images" / "blurred"
EDGE_DIR = HW_DIR / "images" / "edges"
PLOT_DIR = HW_DIR / "images" / "plots"
RESULTS_DIR = HW_DIR / "results"

for folder in [
    CONVERTED_DIR,
    TRANSFORMED_DIR,
    BLURRED_DIR,
    EDGE_DIR,
    PLOT_DIR,
    RESULTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
