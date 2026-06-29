import cv2 
import os 
#Threshold Segmentation

INPUT="images/segmentation/normalized/normalized_rgb.png"
SAVE="images/segmentation"
img=cv2.imread(INPUT) 
gray=cv2.cvtColor( 
    img, cv2.COLOR_BGR2GRAY 
    )

# OTSU
_, otsu = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
otsu_fg = cv2.bitwise_and( img, img, mask=otsu )

 # ADAPTIVE 
adaptive=cv2.adaptiveThreshold( 
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5 
    )
adaptive_fg=cv2.bitwise_and( img, img, mask=adaptive ) 
cv2.imwrite( f"{SAVE}/otsu_mask.png", otsu )
cv2.imwrite( f"{SAVE}/adaptive_mask.png", adaptive )
cv2.imwrite( f"{SAVE}/otsu_segment.png", otsu_fg ) 
cv2.imwrite( f"{SAVE}/adaptive_segment.png", adaptive_fg )
print("Threshold segmentation complete")