import cv2 

import os
# Create Multi-Channel Color Normalization

IMAGE="images/original/HW1_IMG_CS898BA.png" 
SAVE="images/segmentation/normalized"
os.makedirs(SAVE,exist_ok=True)
img=cv2.imread(IMAGE) 
b,g,r=cv2.split(img) 
b=cv2.equalizeHist(b) 
g=cv2.equalizeHist(g) 
r=cv2.equalizeHist(r) 
normalized=cv2.merge([b,g,r]) 
cv2.imwrite( f"{SAVE}/normalized_rgb.png", normalized ) 
print("Normalization complete")