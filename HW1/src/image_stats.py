import cv2
import numpy as np
import pandas as pd
from scipy.stats import mode, skew

from config import IMAGE_PATH, RESULTS_DIR


def main():
    img = cv2.imread(str(IMAGE_PATH))

    if img is None:
        raise FileNotFoundError(f"Image could not be loaded: {IMAGE_PATH}")

    channels = ["Blue", "Green", "Red"]

    results = []
    for i, name in enumerate(channels):
        data = img[:, :, i].flatten()

        results.append({
            "Channel": name,
            "Min": np.min(data),
            "Max": np.max(data),
            "Mean": np.mean(data),
            "Median": np.median(data),
            "Mode": mode(data, keepdims=True).mode[0],
            "Skew": skew(data),
            "Range": np.max(data) - np.min(data),
            "StdDev": np.std(data),
            "Variance": np.var(data),
        })

    df = pd.DataFrame(results)

    print(df)
    df.to_csv(RESULTS_DIR / "image_statistics.csv", index=False)


if __name__ == "__main__":
    main()
