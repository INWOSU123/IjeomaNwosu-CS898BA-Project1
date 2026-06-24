import cv2 
import numpy as np
img=cv2.imread( "images/segmentation/normalized/normalized_rgb.png" ) 
hsv=cv2.cvtColor( 
    img, 
    cv2.COLOR_BGR2HSV 
    ) 
pixels=hsv.reshape(
    (-1,3)
    ) 
pixels=np.float32( 
    pixels 
    )

K=4 

criteria=( 
    cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER,
      100, 0.2 
      )
_,labels,centers=cv2.kmeans( 
    pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS 
    ) 
labels=labels.flatten() 

target=2 
mask=( 
    labels==target 
    ).astype( 
        np.uint8 )*255

mask=mask.reshape( 
    img.shape[:2] 
    ) 

segment=cv2.bitwise_and( 
    img, img, mask=mask 
    )
cv2.imwrite( "images/segmentation/kmeans_mask.png", mask ) 
cv2.imwrite( "images/segmentation/kmeans_segment.png", segment ) 
print("KMeans complete")