import numpy as np
import cv2
from skimage.feature import local_binary_pattern

def lbp_hist(image, keep_ar=False):
    CELL_W, CELL_H = 18, 21
    N_CELLS = 6, 7
    BINS = range(60)
    IMAGE_SIZE = (CELL_W*N_CELLS[0], CELL_H*N_CELLS[1])
    IMAGE_CROP_OFFSET = (IMAGE_SIZE[1]-IMAGE_SIZE[0])//2
    # Convert to grayscale
    if len(image.shape)>=3 and image.shape[2]>1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to target size (keep aspect ratio)
    if keep_ar:
        image = cv2.resize(image, (IMAGE_SIZE[1],IMAGE_SIZE[1]))
        image = image[:,IMAGE_CROP_OFFSET:-IMAGE_CROP_OFFSET-1]
    else:
        image = cv2.resize(image, IMAGE_SIZE)
    # Compute LBP(u2,P=8,R=2) 
    image_lbp = local_binary_pattern(image, 8, 2, method='nri_uniform').astype(np.uint8)
    # Compute and stack histograms over each cell
    feature_vector = []
    for x in range(N_CELLS[0]):
        for y in range(N_CELLS[1]):
            h,_ = np.histogram(image_lbp[y*CELL_H:(y+1)*CELL_H, x*CELL_W:(x+1)*CELL_W], bins=BINS)
            feature_vector += list(h)
    return np.array(feature_vector, dtype=np.uint8), image_lbp

if __name__ == "__main__":
    image = cv2.imread('test.jpg')
    h = lbp_hist(image)
