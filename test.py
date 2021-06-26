import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.model import Model
from options.test_options import TestOptions
from utils.util import center_mask, irregular_mask, save_images

SUPPORTED_IMAGE_TYPES = ["jpg", "png", "jpeg"]


def load_image(image_file, config):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, [config.image_shape[0], config.image_shape[1]])
    image = image / 255.0

    return image


def test(config):
    if config.test_file_path != "":
        file_path = Path(config.test_file_path)
        print(f"Running with single file {file_path}")
        count = 0

        if file_path.suffix.lower()[1:] not in SUPPORTED_IMAGE_TYPES:
            raise Exception(f"File {file_path} is not supported")
        if not file_path.exists():
            raise Exception(f"File {file_path} does not exist")
        if not file_path.is_file():
            raise Exception(f"File {file_path} is not a file")

        print("Processing Image -", file_path)
        if config.random_mask == 1:
            mask = irregular_mask(config.image_shape[0], config.image_shape[1], config.min_strokes, config.max_strokes)
        else:
            mask = center_mask(config.image_shape[0], config.image_shape[1])

        gt_image = load_image(str(file_path), config)
        gt_image = np.expand_dims(gt_image, axis=0)

        input_image = np.where(mask == 1, 1, gt_image)

        prediction_coarse, prediction_refine = generator([input_image, mask], training=False)
        prediction_refine = prediction_refine * mask + gt_image * (1 - mask)

        output_file = str(Path(config.testing_dir).joinpath(file_path.name))
        save_images(input_image[0, ...], gt_image[0, ...], prediction_coarse[0, ...], prediction_refine[0, ...], output_file)

        count += 1
        if count == config.test_num:
            return
        print("-" * 20)
    else:
        count = 0
        print(f"Running with directory {config.test_dir}")
        for root, dirs, files in os.walk(config.test_dir):
            for file in files:
                if not file.split(".")[-1] in ["jpg", "png", "jpeg"]:
                    continue

                print("Processing Image -", file)
                if config.random_mask == 1:
                    mask = irregular_mask(config.image_shape[0], config.image_shape[1], config.min_strokes, config.max_strokes)
                else:
                    mask = center_mask(config.image_shape[0], config.image_shape[1])

                gt_image = load_image(os.path.join(root, file), config)
                gt_image = np.expand_dims(gt_image, axis=0)

                input_image = np.where(mask == 1, 1, gt_image)

                prediction_coarse, prediction_refine = generator([input_image, mask], training=False)
                prediction_refine = prediction_refine * mask + gt_image * (1 - mask)
                save_images(input_image[0, ...], gt_image[0, ...], prediction_coarse[0, ...], prediction_refine[0, ...], os.path.join(config.testing_dir, file))

                count += 1
                if count == config.test_num:
                    return
                print("-" * 20)


if __name__ == "__main__":
    # Loading the arguments
    config = TestOptions().parse()

    model = Model()
    generator = model.build_generator()

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(os.path.join(config.pretrained_model_dir, config.checkpoint_prefix))

    test(config)
