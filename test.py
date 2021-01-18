import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import datetime

from models.model import Model
from utils.util import *
from options.test_options import TestOptions

from tqdm import tqdm
from PIL import Image, ImageDraw

def load_image (image_file, config) :
    image = tf.io.read_file (image_file) 
    image = tf.image.decode_jpeg (image, channels=3)
    image = tf.cast (image, dtype=tf.float32)
    image = tf.image.resize (image, [config.image_shape[0], config.image_shape[1]])
    image = image / 255.0

    return image

def test (config) :
    if config.random_mask == 0 :
        mask = center_mask (config.image_shape[0], config.image_shape[1])

        count = 0
        for root, dirs, files in os.walk (config.test_dir) :
            for file in files :
                if not file.split('.')[-1] in ['jpg', 'png', 'jpeg'] :
                    continue
                
                gt_image = load_image (file)
                input_image = np.where (mask==1, 1, gt_image)

                input_image = np.expand_dims (input_image, axis=0)
                
                prediction_coarse, prediction_refine = generator ([img, mask], training=False)
                prediction_refine = prediction_refine * mask + gt_image * (1  - mask)
                save_images (input_image, gt_image, prediction_coarse, prediction_refine, os.path.join (config.testing_dir, file))
                
                count += 1
                if count == config.test_num :
                    return
    else :
        count = 0
        for root, dirs, files in os.walk (config.test_dir) :
            for file in files :
                if not file.split('.')[-1] in ['jpg', 'png', 'jpeg'] :
                    continue
                
                mask = irregular_mask (config.image_shape[0], config.image_shape[1])
                gt_image = load_image (file)
                input_image = np.where (mask==1, 1, gt_image)

                input_image = np.expand_dims (input_image, axis=0)
                
                prediction_coarse, prediction_refine = generator ([img, mask], training=False)
                prediction_refine = prediction_refine * mask + gt_image * (1  - mask)
                save_images (input_image, gt_image, prediction_coarse, prediction_refine, os.path.join (config.testing_dir, file))
                
                count += 1
                if count == config.test_num :
                    return

if __name__ == '__main__' :
    # Loading the arguments
    config = TestOptions().parse ()

    model = Model ()
    generator = model.build_generator ()

    checkpoint = tf.train.Checkpoint (generator=generator)
    checkpoint.restore (os.path.join (config.pretrained_model_dir, config.checkpoint_prefix))

    test (config)