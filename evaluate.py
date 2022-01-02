import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

PATHS = []
COUNT_IMAGES = 1

print(PATHS)

for PATH in PATHS:
    for root, dirs, files in os.walk(PATH):
        print(PATH)
        ssim = np.zeros((COUNT_IMAGES, 1))
        psnr = np.zeros((COUNT_IMAGES, 1))
        l1 = np.zeros((COUNT_IMAGES, 1))
        l2 = np.zeros((COUNT_IMAGES, 1))

        images = []

        for file in tqdm(files):
            image = tf.convert_to_tensor(np.array(Image.open(root + file)) / 255.0, dtype=tf.float32)

            input_image = image[:, :256, :]
            ground_truth = image[:, 256:512, :]

            for i in range(COUNT_IMAGES):
                output_image = image[:, (i + 3) * 256 : (i + 4) * 256, :]
                ssim[i] += tf.image.ssim_multiscale(output_image, ground_truth, max_val=1.0)
                psnr[i] += tf.image.psnr(output_image, ground_truth, max_val=1.0)
                l1[i] += tf.math.reduce_mean(tf.math.abs(output_image - ground_truth))
                l2[i] += tf.reduce_mean((output_image - ground_truth) ** 2)

        print(ssim / len(files))
        print(psnr / len(files))
        print(l1 / len(files))
        print(l2 / len(files))
