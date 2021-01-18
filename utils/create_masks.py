# Generate Irregular Random Masks of random percentages 

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def irregular_shape (mask) :
    mask_shape = mask.shape
    model = random.random()
    if model < 0.6:
        # Draw random lines
        x1, x2 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[0])
        y1, y2 = random.randint(1, mask_shape[1]), random.randint(1, mask_shape[1])
        thickness = random.randint(4, MAX_WIDTH)
        cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

    elif model > 0.6 and model < 0.8:
        # Draw random circles
        x1, y1 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
        radius = random.randint(4, MAX_WIDTH)
        cv2.circle(mask, (x1, y1), radius, (1, 1, 1), -1)

    elif model > 0.8:
        # Draw random ellipses
        x1, y1 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
        s1, s2 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
        a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
        thickness = random.randint(4, MAX_WIDTH)
        cv2.ellipse(mask, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)
            
    return mask

def reduce_mask (mask, count) :
    x, y = np.where (mask == 1)
    for i in range (count) :
        mask[x[i], y[i]] = 0
        
    return mask

COUNT = 2000

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MAX_WIDTH = 20

mask_dirs = [(0.5, 0.6, "Random_masks_50-60/"), ]

for MIN_PERCENT, MAX_PERCENT, mask_dir in mask_dirs :

    if not os.path.isdir (mask_dir) :
        os.mkdir (mask_dir)

    for i in range (COUNT) :
        mask = np.zeros ([IMAGE_HEIGHT, IMAGE_WIDTH], np.uint8)
        while (True) :
            if float (np.sum (mask) / (IMAGE_HEIGHT * IMAGE_WIDTH)) >= MIN_PERCENT and float (np.sum (mask) / (IMAGE_HEIGHT * IMAGE_WIDTH)) <= MAX_PERCENT :
                break
            elif float (np.sum (mask) / (IMAGE_HEIGHT * IMAGE_WIDTH)) > MAX_PERCENT :
                mask = reduce_mask (mask, int (np.sum (mask) - MAX_PERCENT * (IMAGE_HEIGHT * IMAGE_WIDTH)))
            
            mask = irregular_shape (mask)
            
        plt.imsave (mask_dir + "mask" + str(i) + ".jpg", mask, cmap='gray')