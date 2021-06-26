import tensorflow as tf

from models.gc_layer import GatedConvolution, GatedDeConvolution
from models.hypergraph_layer import HypergraphConv


class Model:
    # Generator Network
    def build_generator(self):
        input_img = tf.keras.layers.Input(shape=[256, 256, 3])
        input_mask = tf.keras.layers.Input(shape=[256, 256, 1])

        # Coarse network
        c_num = 64

        x = tf.keras.layers.Concatenate()([input_img, input_mask])

        x = GatedConvolution(channels=c_num, kernel_size=7, stride=1, dilation=1, padding="same", activation="ELU")(x)

        x = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=2, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        skip_coarse = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)

        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=2, dilation=1, padding="same", activation="ELU")(skip_coarse)
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)

        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=2, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=2, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=2, padding="same", activation="ELU")(x)

        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)

        x = GatedDeConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = tf.keras.layers.Concatenate()([x, skip_coarse])
        x = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)

        x = GatedDeConvolution(channels=c_num, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        coarse_out = GatedConvolution(channels=3, kernel_size=3, stride=1, dilation=1, padding="same", activation=None)(x)

        # Refine Net

        x = coarse_out * input_mask + input_img * (1 - input_mask)
        x = tf.keras.layers.Concatenate()([x, input_mask])
        x = GatedConvolution(channels=c_num, kernel_size=7, stride=1, dilation=1, padding="same", activation="ELU")(x)

        x = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=2, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        skip_1 = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)

        # 128, 128, 64

        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=2, dilation=1, padding="same", activation="ELU")(skip_1)
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)

        # First Hypergraph conovlutuion layer
        skip_2 = HypergraphConv(
            in_features=128, out_features=128, features_height=64, features_width=64, edges=256, filters=128, apply_bias=True, trainable=True
        )(x)
        skip_2 = tf.keras.layers.ELU()(skip_2)

        # 64, 64, 128

        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=2, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=1, dilation=2, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=1, dilation=2, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=1, dilation=2, padding="same", activation="ELU")(x)

        # Second Hypergraph convolution layer
        skip_3 = HypergraphConv(
            in_features=256, out_features=512, features_height=32, features_width=32, edges=256, filters=128, apply_bias=True, trainable=True
        )(x)
        skip_3 = tf.keras.layers.ELU()(skip_3)

        # 32, 32, 256

        x = GatedConvolution(channels=c_num * 16, kernel_size=3, stride=2, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 16, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 16, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)

        # 16, 16, 512

        x = GatedDeConvolution(channels=c_num * 8, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = tf.keras.layers.Concatenate()([x, skip_3])
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=1, dilation=2, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=1, dilation=2, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=1, dilation=2, padding="same", activation="ELU")(x)

        # 32, 32, 256

        x = GatedDeConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = tf.keras.layers.Concatenate()([x, skip_2])
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)

        # 64, 64, 128

        x = GatedDeConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = tf.keras.layers.Concatenate()([x, skip_1])
        x = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)

        # 128, 128, 64

        x = GatedDeConvolution(channels=c_num, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        x = GatedConvolution(channels=c_num, kernel_size=3, stride=1, dilation=1, padding="same", activation="ELU")(x)
        refine_out = GatedConvolution(channels=3, kernel_size=3, stride=1, dilation=1, padding="same", activation=None)(x)

        return tf.keras.Model(inputs=[input_img, input_mask], outputs=[coarse_out, refine_out])

    # Discriminator Network
    def build_discriminator(self):
        input_img = tf.keras.layers.Input(shape=[256, 256, 3])
        input_mask = tf.keras.layers.Input(shape=[256, 256, 1])

        x = tf.keras.layers.Concatenate()([input_img, input_mask])

        c_num = 64
        x = GatedConvolution(channels=c_num, kernel_size=3, stride=1, dilation=1, padding="same", activation="LeakyReLU")(x)
        x = GatedConvolution(channels=c_num * 2, kernel_size=3, stride=2, dilation=1, padding="same", activation="LeakyReLU")(x)
        x = GatedConvolution(channels=c_num * 4, kernel_size=3, stride=2, dilation=1, padding="same", activation="LeakyReLU")(x)
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=2, dilation=1, padding="same", activation="LeakyReLU")(x)
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=2, dilation=1, padding="same", activation="LeakyReLU")(x)
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=2, dilation=1, padding="same", activation="LeakyReLU")(x)
        x = GatedConvolution(channels=c_num * 8, kernel_size=3, stride=2, dilation=1, padding="same", activation="LeakyReLU")(x)

        return tf.keras.Model(inputs=[input_img, input_mask], outputs=x)
