import cv2
import numpy as np
from scipy.stats import skew, mode
import pandas as pd

from config import IMAGE_PATH

img = cv2.imread(IMAGE_PATH)

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
        "Range": np.max(data)-np.min(data),
        "StdDev": np.std(data),
        "Variance": np.var(data)
    })

df = pd.DataFrame(results)

print(df)

df.to_csv("image_statistics.csv", index=False)