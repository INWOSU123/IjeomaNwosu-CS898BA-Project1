# IjeomaNwosu-CS898BA-Project1

 Visual Statistics
The majority of the pixel values are focused in the darker intensity regions, indicating a significant underexposure. A non-uniform distribution that is compatible with nighttime imagery is indicated by the skewness values.
Equalization in Histograms
Without considerably changing color information, equalizing the V channel in HSV greatly increased visibility of the foreground figure and surrounding structures.
Blur Gaussian
Noise and fine details were gradually reduced as sigma values increased. Larger sigma values resulted in smoother images at the expense of edge sharpness, whereas low sigma values maintained edges.
Analysis of Edge Detection
Sobel
• Accurate directional edge data
• Moderately sensitive to noise
Laplacian
• Captures a lot of edges
• In dark photos, it generates noisy outputs.
Prewitt
• Easy to compute
• Inaccurate edge localization
Canny
• The ideal ratio of edge detection to noise reduction
• Created the figure's most distinct outline
Concluding Remarks
Because of its non-maximum suppression and hysteresis thresholding, which produced cleaner object boundaries and fewer false edges than the other methods, Canny edge detection worked best for this doorbell-camera image
