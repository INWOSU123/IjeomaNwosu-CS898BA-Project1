from pathlib import Path

HW_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HW_DIR.parent

IMAGE_PATH = PROJECT_ROOT / "images" / "original" / "HW1_IMG_CS898BA.png"

NORMALIZED_DIR = HW_DIR / "images" / "normalized"
MASK_DIR = HW_DIR / "images" / "masks"
SEGMENT_DIR = HW_DIR / "images" / "segments"
COMPARISON_DIR = HW_DIR / "images" / "comparison"

NORMALIZED_IMAGE = NORMALIZED_DIR / "normalized_rgb.png"
REFERENCE_MASK = MASK_DIR / "reference_mask.png"

for folder in [
    NORMALIZED_DIR,
    MASK_DIR,
    SEGMENT_DIR,
    COMPARISON_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
