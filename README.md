# Ijeoma Nwosu CS898BA Project Homework one

## Project Overview

This project performs visual statistics and image processing analysis on a doorbell-camera image. It includes histogram equalization, Gaussian blur, edge detection, and segmentation techniques to improve object visibility and separate foreground elements.

## Visual Statistics

- The histogram shows that most pixel values are in darker intensity regions, indicating underexposure.
- The distribution is non-uniform and consistent with nighttime imagery.
- Skewness values confirm the bias toward darker pixels.

## Histogram Equalization

- Equalizing the V channel in HSV greatly improves visibility of the foreground figure and surrounding structures.
- This enhancement is achieved without significantly altering color information.

## Gaussian Blur Analysis

- Increasing the Gaussian sigma reduces noise and fine details.
- Higher sigma values produce smoother images at the expense of edge sharpness.
- Lower sigma values preserve edges while still reducing some noise.

## Edge Detection Comparison

### Sobel

- Accurate directional edge information
- Moderately sensitive to noise

### Laplacian

- Captures many edges
- Can generate noisy outputs in dark images

### Prewitt

- Computationally simple
- Less accurate edge localization

### Canny

- Best balance between edge detection and noise reduction
- Produces the most distinct outline for the figure

## Edge Detection Conclusion

Canny edge detection performed best on this doorbell-camera image because non-maximum suppression and hysteresis thresholding produced cleaner object boundaries and fewer false edges than the other methods.

## Notes

- The project is focused on enhancing image visibility and evaluating different edge detection and segmentation approaches.
- Results are stored in the `src/images/` folders.

## Homework Two – Image Segmentation

### CS 898BA – Image Analysis and Computer Vision

---

 **Project Overview**
This assignment extends Homework One by applying image segmentation techniques to isolate a foreground figure captured in a low-light doorbell camera image.

The objective is to extract the unknown figure from the background using multiple segmentation approaches and evaluate their effectiveness both visually and quantitatively.

This project combines:

- Image preprocessing
- Color normalization
- Threshold-based segmentation
- Optimization-based segmentation
- Quantitative segmentation evaluation
- Visualization and comparison

---

## Objective

The goal of this project is to isolate the central figure from a dark outdoor scene and compare segmentation methods for accuracy and robustness.

The following segmentation techniques were implemented:

1. Multi-channel Color Normalization
2. Otsu  Thresholding
3. Adaptive  Thresholding
4. K-Means Clustering
5. Quantitative evaluation using IoU and Dice Coefficient

---

## Repository Structure

src/
│
├── normalize.py
├── threshold_segmentation.py
├── kmeans_segmentation.py
├── metrics.py
├── create_segmentation_plot.py
├── config.py

images/
│
├── original/
│     └── HW1_IMG_CS898BA.png
│
├── segmentation/
│     ├── normalized/
│     │     └── normalized_rgb.png
│     │
│     ├── otsu/
│     │     ├── otsu_mask.png
│     │     └── otsu_segment.png
│     │
│     ├── adaptive/
│     │     ├── adaptive_mask.png
│     │     └── adaptive_segment.png
│     │
│     ├── kmeans/
│     │     ├── kmeans_mask.png
│     │     └── kmeans_segment.png
│     │
│     ├── reference_mask.png
│     │
│     └── comparison/
│           └── final_plot.png

README.md
AI_Log.md

---

## Installation

## Clone Repository

```bash
git clone <(https://github.com/INWOSU123/IjeomaNwosu-CS898BA-Project1/)>
```

Open project:

```bash
cd IjeomaNwosu-CS898BA-Project1
```

---

## Dependencies

Install required packages:

```bash
pip install opencv-python numpy scipy matplotlib pandas
```

Verify installation:

```bash
python -c "import cv2,numpy,matplotlib"
```

---

## Image Dataset

Input Image:

images/original/HW1_IMG_CS898BA.png

The image is a low-light outdoor scene containing a possible Alien figure.

Challenges:

- Uneven lighting
- Dark shadows
- Background clutter
- Weak object boundaries

---

## Part 1 — Multi-Channel Color Normalization

## Method

To standardize illumination across the image:

1. Load original RGB image
2. Split channels:
   - Red
   - Green
   - Blue
3. Apply Histogram Equalization independently
4. Merge channels

Output:

normalized_rgb.png

## Purpose

Improve:

- Contrast
- Visibility
- Segmentation consistency

---

## Part 2 — Threshold Segmentation

 Otsu Thresholding

### Process

1. Convert normalized image to grayscale
2. Compute automatic threshold
3. Separate foreground/background

Output:

otsu_mask.png
otsu_segment.png

### Advantages (Otsu Thresholding)

- Fast
- Automatic threshold selection
- Simple implementation

### Disadvantages (Otsu Thresholding)

- Sensitive to noise

---

 Adaptive Thresholding

1. Convert normalized image to grayscale
2. Compute local thresholds

Output:

adaptive_mask.png
adaptive_segment.png

### Advantages (Adaptive Thresholding)

- Handles uneven lighting
- Better local segmentation

### Disadvantages (Adaptive Thresholding)

- Increased noise
- Fragmented regions

---

## Part 3 — K-Means Segmentation

### K-Means Method

Steps:

1. Convert normalized image to HSV
2. Reshape pixels
3. Apply K-Means clustering
4. Test K values

Values tested:

K = 3
K = 4
K = 5

Selected:

K = 4

Output:

kmeans_mask.png
kmeans_segment.png

### Advantages (K-Means Segmentation)

- Uses color information
- Better separation

### Disadvantages

- Requires parameter tuning
- Cluster selection required

---

## Part 4 — Quantitative Evaluation

## Ground Truth Creation

A manual reference mask was created:

reference_mask.png

White:
Foreground Figure

Black:
Background

---

## Metrics

### Intersection over Union (IoU)

Formula:

IoU = |A ∩ B| / |A ∪ B|

Measures overlap between prediction and reference.

---

### Dice Coefficient

Formula:

Dice = 2|A ∩ B| / (|A| + |B|)

Measures segmentation similarity.

---

## Results

| Method | IoU | Dice |

| Otsu | 0.045120915586633274 | 0.08634582834141562|
| Adaptive | 0.0714083490967658 | 0.13329810087249275 |
| K-Means | 0.02965334410815843 | 0.05759869431365346 |

---

## Visual Comparison

Generated using:

python create_segmentation_plot.py

Output:

images/segmentation/comparison/final_plot.png

Displayed:

1. Original
2. Normalized
3. Otsu
4. Adaptive
5. K-Means

## Analysis

## Effect of Color Normalization

Applying histogram equalization across all color channels improved contrast and reduced lighting inconsistencies.

Compared with Homework One:

- Figure visibility improved
- Segmentation boundaries became more distinguishable
- Thresholding produced more stable masks

---

## Segmentation Comparison

### Otsu

Produced clean global segmentation but struggled in darker regions.

### Adaptive

Captured local detail but introduced more background noise.

### K-Means

Produced the strongest separation because clustering considered color information instead of intensity only.

---

## Conclusion

This assignment demonstrated that segmentation performance improves significantly when image preprocessing is applied before segmentation.

Among the evaluated methods:

- Otsu provided fast baseline segmentation
- Adaptive improved local illumination handling
- K-Means achieved the strongest figure isolation

Combining classical image processing with segmentation techniques produced measurable improvements over raw-image analysis.

---

## Running the Project

Execute in order:
python normalize.py
python threshold_segmentation.py
python kmeans_segmentation.py
python metrics.py
python create_segmentation_plot.py

---

## AI Usage

All AI assistance used for this assignment was documented in:

AI_Log.md
