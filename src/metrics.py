import cv2 
import numpy as np

#Quantitative Evaluation using IoU and Dice Coefficient

truth=cv2.imread( "images/segmentation/reference_mask.png", 0 ) 
methods=[ "otsu_mask.png", "adaptive_mask.png", "kmeans_mask.png" ] 
for f in methods: 
    pred=cv2.imread( f"images/segmentation/{f}", 0 ) 
    truth=truth>0 
    pred=pred>0 
    inter=np.logical_and( truth, pred ).sum() 
    union=np.logical_or( truth, pred ).sum() 
    iou=inter/union 
    dice=( 2*inter )/( truth.sum()+ pred.sum() ) 
    print(f) 
    print("IoU:",iou)
    print("Dice:",dice)