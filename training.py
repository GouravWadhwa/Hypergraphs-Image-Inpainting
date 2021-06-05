import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
import datetime

from tqdm import tqdm

from models.model import Model
from utils.util import *
from utils.losses import *
from options.train_options import TrainOptions

def load_images (image_file) :
    original_image = tf.io.read_file (image_file) 
    original_image = tf.image.decode_jpeg (original_image, channels=3)
    original_image = tf.cast (original_image, dtype=tf.float32)
    original_image = tf.image.resize (original_image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    original_image = original_image / 255.0

    return original_image

def predict (config, dataset, epoch, mask_function) :
    for n, (original_image) in dataset.take(100).enumerate() :
        mask = mask_function (config.image_shape[0], config.image_shape[1])
        masked_image = np.where (mask==1, 1, original_image)

        prediction_coarse, prediction_refine = generator ([masked_image, mask], training=False)
        
        path = os.path.join (config.training_dir, f"EPOCH{epoch}", f"Image{n}.jpg")
        print (path)
        save_images (masked_image[0, :, :, :], original_image[0, :, :, :], prediction_coarse[0, :, :, :], prediction_refine[0, :, :, :], path)

def generator_loss (disc_generated_output, gen_output_coarse, gen_output_refine, target, mask) :
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gan_loss = loss_object (tf.ones_like (disc_generated_output), disc_generated_output)

    hole_l1_loss = tf.reduce_mean (tf.math.abs ((mask) * (target - gen_output_coarse))) * 0.5
    hole_l1_loss += tf.reduce_mean (tf.math.abs ((mask) * (target - gen_output_refine)))

    valid_l1_loss = tf.reduce_mean (tf.math.abs ((1 - mask) * (target - gen_output_coarse))) * 0.5
    valid_l1_loss += tf.reduce_mean (tf.math.abs ((1 - mask) * (target - gen_output_refine)))
    
    vgg_gen_output = vgg_model (tf.keras.applications.vgg19.preprocess_input (gen_output_refine * 255.0))
    vgg_comp = vgg_model (tf.keras.applications.vgg19.preprocess_input ((gen_output_refine * mask + target * (1-mask)) * 255.0))
    vgg_target = vgg_model (tf.keras.applications.vgg19.preprocess_input (target * 255.0))
    perceptual_loss_out = 0
    perceptual_loss_comp = 0

    for i in range (len (selected_layers)) :
        perceptual_loss_out += tf.reduce_mean (tf.math.abs(vgg_gen_output[i] - vgg_target[i]))
        perceptual_loss_comp += tf.reduce_mean (tf.math.abs(vgg_comp[i] - vgg_target[i]))

    edge_loss = tf.reduce_mean(tf.math.abs (tf.image.sobel_edges (gen_output_refine) - tf.image.sobel_edges (target)))

    total_loss = VALID_LOSS_WEIGHT * valid_l1_loss + HOLE_LOSS_WEIGHT * hole_l1_loss + EDGE_LOSS_WEIGHT * edge_loss + GAN_LOSS_WEIGHT * gan_loss + PERCEPTUAL_LOSS_OUT_WEIGHT * perceptual_loss_out + PERCEPTUAL_LOSS_COMP_WEIGHT * perceptual_loss_comp
    
    return total_loss, valid_l1_loss, hole_l1_loss, edge_loss, gan_loss, perceptual_loss_out, perceptual_loss_comp

def discriminator_loss (disc_original_output, disc_generated_output) :
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    original_loss = loss_object (tf.ones_like (disc_original_output), disc_original_output)
    generated_loss = loss_object (tf.zeros_like (disc_generated_output), disc_generated_output)
    
    total_loss = original_loss + generated_loss
    
    return total_loss

def train_step (original_image, masked_image, mask, epoch) :
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape :
        prediction_coarse, prediction_refine = generator ([masked_image, mask], training=True)

        disc_original_output = discriminator ([original_image, mask], training=True)
        disc_generated_output = discriminator ([prediction_refine, mask], training=True)

        total_loss, valid_l1_loss, hole_l1_loss, edge_loss, gan_loss, pl_out, pl_comp = generator_loss (disc_generated_output, prediction_coarse, prediction_refine, original_image, mask)
        disc_loss = discriminator_loss (disc_generated_output=disc_generated_output, disc_original_output=disc_original_output)

    gen_gradients = gen_tape.gradient (total_loss, generator.trainable_variables) 
    disc_gradients = disc_tape.gradient (disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients (zip (gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients (zip (disc_gradients, discriminator.trainable_variables))

    return total_loss, valid_l1_loss, hole_l1_loss, edge_loss, gan_loss, pl_out, pl_comp, disc_loss

def fit (config, train_dataset, epochs) :
    mask_function = None
    if config.random_mask == 0 :
        mask_function = center_mask
    else :
        mask_function = irregular_mask

    for epoch in range (0, epochs) :
        if not os.path.isdir (os.path.join (config.training_dir, f"EPOCH{epoch}")) :
            os.mkdir (os.path.join (config.training_dir, f"EPOCH{epoch}"))

        avg_total_loss = 0
        avg_valid_l1_loss = 0
        avg_hole_l1_loss = 0
        avg_edge_loss = 0
        avg_gan_loss = 0
        avg_pl_out = 0
        avg_pl_comp = 0
        avg_disc_loss = 0

        print ("EPOCH : " + str (epoch))
        for n, (original_image) in tqdm (train_dataset.enumerate()) :
            mask = mask_function (config.image_shape[0], config.image_shape[1])
            masked_image = np.where (mask==1, 1, original_image)

            total_loss, valid_l1_loss, hole_l1_loss, edge_loss, gan_loss, pl_out, pl_comp, disc_loss = train_step (original_image, masked_image, mask, epoch)
            
            avg_total_loss += total_loss
            avg_valid_l1_loss += valid_l1_loss
            avg_hole_l1_loss += hole_l1_loss
            avg_edge_loss += edge_loss
            avg_gan_loss += gan_loss
            avg_pl_out += pl_out
            avg_pl_comp += pl_comp
            avg_disc_loss += disc_loss

        avg_total_loss /= n.numpy() + 1
        avg_valid_l1_loss /= n.numpy() + 1
        avg_hole_l1_loss /= n.numpy() + 1
        avg_edge_loss /= n.numpy() + 1
        avg_gan_loss /= n.numpy() + 1
        avg_pl_out /= n.numpy() + 1
        avg_pl_comp /= n.numpy() + 1
        avg_disc_loss /= n.numpy() + 1

        print ('avg_total_loss = ', avg_total_loss)
        print ('avg_valid_l1_loss = ', avg_valid_l1_loss)
        print ('avg_hole_l1_loss = ', avg_hole_l1_loss)
        print ('avg_edge_loss = ', avg_edge_loss)
        print ('avg_gan_loss = ', avg_gan_loss)
        print ('avg_pl_out = ', avg_pl_out)
        print ('avg_pl_comp = ', avg_pl_comp)
        print ('avg_disc_loss = ', avg_disc_loss)

        checkpoint.save (file_prefix=checkpoint_prefix)
        predict (config, train_dataset, epoch, mask_function)

def train (config) :
    train_files = []
    if config.train_file_path == '' :
        for root, dirs, files in os.walk (config.train_dir) :
            for file in files :
                train_files.append (os.path.join (root, file))
    else :
        train_files = open (config.train_file_path)
        train_files = [line[:-1] for line in train_files.readlines()]

    train_dataset = tf.data.Dataset.from_tensor_slices (train_files)
    train_dataset = train_dataset.map (load_images)
    train_dataset = train_dataset.shuffle (BUFFER_SIZE).batch (BATCH_SIZE)

    fit (config, train_dataset, config.epochs)

if __name__ == '__main__' :
    config = TrainOptions ().parse ()

    HOLE_LOSS_WEIGHT = config.hole_l1_loss 
    VALID_LOSS_WEIGHT = config.valid_l1_loss
    EDGE_LOSS_WEIGHT = config.edge_loss
    GAN_LOSS_WEIGHT = config.gan_loss
    PERCEPTUAL_LOSS_COMP_WEIGHT = config.pl_comp
    PERCEPTUAL_LOSS_OUT_WEIGHT = config.pl_out

    IMAGE_HEIGHT = config.image_shape[0]
    IMAGE_WIDTH = config.image_shape[1]

    BUFFER_SIZE = config.buffer_size 
    BATCH_SIZE = config.batch_size

    EPOCHS = config.epochs

    learning_rate = config.learning_rate
    decay_steps = config.decay_steps
    decay_rate = config.decay_rate

    RESTORE_CHECKPOINT = False
    if config.pretrained_model_dir != '' :
        RESTORE_CHECKPOINT = True

    # Developing the VGG Model for the Perceptual Loss
    vgg = tf.keras.applications.VGG19 (include_top=False, weights='imagenet')
    vgg.trainable = False

    selected_layers = [
        'block3_conv4',
        'block4_conv4',
        'block5_conv3',
        'block5_conv4'
    ]

    outputs = [vgg.get_layer(name).output for name in selected_layers]
    vgg_model = tf.keras.Model ([vgg.input], outputs)

    model = Model ()
    generator = model.build_generator ()
    discriminator = model.build_discriminator ()

    learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay (
        initial_learning_rate=config.learning_rate, 
        decay_steps=config.decay_steps, 
        decay_rate=config.decay_rate,
        staircase=True
    )

    discriminator_optimizer = tf.keras.optimizers.Adam (learning_rate=learning_schedule)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)

    checkpoint = tf.train.Checkpoint (
        generator=generator,
        discriminator=discriminator
    )
    checkpoint_directory = config.checkpoint_saving_dir
    if not os.path.isdir (checkpoint_directory) :
        os.mkdir (checkpoint_directory)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    # Loading the latest checkpoint in this directory
    checkpoint.restore (tf.train.latest_checkpoint (checkpoint_directory))

    if RESTORE_CHECKPOINT :
        # If want to use the latest checkpoint then don't use the pretrained model argument
        checkpoint.restore (os.path.join (config.pretrained_model_dir, config.checkpoint_prefix))

    train (config)
