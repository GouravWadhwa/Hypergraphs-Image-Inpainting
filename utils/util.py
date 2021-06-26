import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


def irregular_mask(image_height, image_width, batch_size=1, min_strokes=16, max_strokes=48):
    masks = []

    for b in range(batch_size):
        mask = np.zeros((image_height, image_width), np.uint8)
        mask_shape = mask.shape

        max_width = 20
        number = random.randint(min_strokes, max_strokes)
        for _ in range(number):
            model = random.random()
            if model < 0.6:
                # Draw random lines
                x1, x2 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[0])
                y1, y2 = random.randint(1, mask_shape[1]), random.randint(1, mask_shape[1])
                thickness = random.randint(4, max_width)
                cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

            elif model > 0.6 and model < 0.8:
                # Draw random circles
                x1, y1 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
                radius = random.randint(4, max_width)
                cv2.circle(mask, (x1, y1), radius, (1, 1, 1), -1)

            elif model > 0.8:
                # Draw random ellipses
                x1, y1 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
                s1, s2 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
                a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
                thickness = random.randint(4, max_width)
                cv2.ellipse(mask, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

        masks.append(mask[:, :, np.newaxis])

    return np.array(masks).astype("float32")


def center_mask(image_height, image_width, batch_size=1):
    mask = np.zeros((batch_size, image_height, image_width, 1)).astype("float32")
    mask[:, image_height // 4 : (image_height // 4) * 3, image_height // 4 : (image_height // 4) * 3, :] = 1.0

    return mask


def save_images(input_image, ground_truth, prediction_coarse, prediction_refine, path):

    display_list = [input_image, ground_truth, prediction_coarse, prediction_refine]
    img = np.concatenate(display_list, axis=1)
    plt.imsave(path, np.clip(img, 0, 1.0))
