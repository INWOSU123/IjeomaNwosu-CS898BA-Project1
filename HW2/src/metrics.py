import cv2
import numpy as np

from config import MASK_DIR, REFERENCE_MASK


def main():
    truth_img = cv2.imread(str(REFERENCE_MASK), 0)

    if truth_img is None:
        raise FileNotFoundError(f"Reference mask could not be loaded: {REFERENCE_MASK}")

    truth = truth_img > 0
    methods = ["otsu_mask.png", "adaptive_mask.png", "kmeans_mask.png"]

    for filename in methods:
        mask_path = MASK_DIR / filename
        pred_img = cv2.imread(str(mask_path), 0)

        if pred_img is None:
            raise FileNotFoundError(f"Prediction mask could not be loaded: {mask_path}")

        pred = pred_img > 0
        inter = np.logical_and(truth, pred).sum()
        union = np.logical_or(truth, pred).sum()
        iou = inter / union if union else 0
        dice = (2 * inter) / (truth.sum() + pred.sum()) if truth.sum() + pred.sum() else 0

        print(filename)
        print("IoU:", iou)
        print("Dice:", dice)


if __name__ == "__main__":
    main()
