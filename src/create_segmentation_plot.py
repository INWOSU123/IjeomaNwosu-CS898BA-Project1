import matplotlib.pyplot as plt 
files=[ "images/original/HW1_IMG_CS898BA.png", 
        "images/segmentation/normalized/normalized_rgb.png",
        "images/segmentation/otsu_mask.png", 
        "images/segmentation/adaptive_mask.png",
        "images/segmentation/kmeans_mask.png" ] 
titles=[ "Original", "Normalized", "Otsu", "Adaptive", "KMeans" ] 
fig,ax=plt.subplots( 

    1, 5, figsize=(22,5) 
    
    ) 
for i,p in enumerate(files): 
    print("Loading:", p)
    img=plt.imread(p) 
    print("Shape:", img.shape)
    ax[i].imshow( img, cmap="gray" ) 
    ax[i].set_title( titles[i] ) 
    ax[i].axis("off") 
plt.tight_layout() 
plt.savefig( "images/segmentation/comparison/final_plot.png" ) 
plt.show()