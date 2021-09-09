import tensorflow as tf

from models.gc_layer import GatedConvolution, GatedDeConvolution
from models.hypergraph_layer import HypergraphConv

class Model () :
    def build_generator (self, input_height, input_width) :
        input_img = tf.keras.layers.Input (shape=(input_height, input_width, 3))
        input_mask = tf.keras.layers.Input (shape=(input_height, input_width, 1))

        channels = 64

        # Coarse Network
        
        x = tf.keras.layers.Concatenate () ([input_img, input_mask])
        x = GatedConvolution (
            channels=channels,
            kernel_size=7,
            stride=1,
            dilation=1,
            padding='same',
            activation='ELU'
        ) (x)

        # Encoder For Coarse Network
        skip_connections = []
        downsamples = 3
        for i in range (1, downsamples) :
            x = GatedConvolution (channels=(2**i)*channels, kernel_size=3, stride=2, dilation=1, padding='same', activation='ELU') (x)
            for j in range (2) :
                x = GatedConvolution (channels=(2**i)*channels, kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU') (x)

            if i != downsamples - 1 :
                skip_connections.append (x)

        # Center Convolutions for higher receptive field
        # These convolutions are with dilation=2
        for i in range (3) :
            x = GatedConvolution (channels=(2**(downsamples-1))*channels, kernel_size=3, stride=1, dilation=2, padding='same', activation='ELU') (x)

        # Decoder Network for Coarse Network
        current = len (skip_connections) - 1
        for i in range (downsamples-1, 0, -1) :
            for j in range (2) :
                x = GatedConvolution (channels=channels*(2**i), kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU') (x)
            
            x = GatedDeConvolution (channels=channels*(2**(i-1)), kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU') (x)
            if current != -1 :
                x = tf.keras.layers.Concatenate () ([x, skip_connections[current]])
                current -= 1

        x = GatedConvolution (channels=channels, kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU') (x)
        coarse_out = GatedConvolution (channels=3, kernel_size=3, stride=1, dilation=1, padding='same', activation=None) (x)
        
        # Refine Network

        x = coarse_out * input_mask + input_img * (1 - input_mask)
        x = tf.keras.layers.Concatenate () ([x, input_mask])
        x = GatedConvolution (
            channels=channels,
            kernel_size=7,
            stride=1,
            dilation=1,
            padding='same',
            activation='ELU'
        ) (x)

        # Encoder For Refine Network
        skip_connections = []
        downsamples = 4
        current_image_height = input_height
        current_image_width = input_width
        for i in range (1, downsamples+1) :
            x = GatedConvolution (channels=channels * (2**i), kernel_size=3, stride=2, dilation=1, padding='same', activation='ELU') (x)
            current_image_height = current_image_height // 2
            current_image_width = current_image_width // 2

            dilation_rate = 1
            count = 2
            if i == downsamples - 1 :
                dilation_rate = 2
                count = 3
            
            for j in range (count) :
                x = GatedConvolution (channels=channels*(2**i), kernel_size=3, stride=1, dilation=dilation_rate, padding='same', activation='ELU') (x)

            if i != downsamples :
                skip_connections.append (list([x, channels*(2**(i-1)), current_image_height, current_image_width]))

        # Apply Hypergraph convolution on last skip connections
        for i in range (len(skip_connections)-1, len(skip_connections)-3, -1) :
            mult = 1
            if i == len(skip_connections)-1 :
                mult = 2
            skip_connections[i][0] = HypergraphConv(
                in_features=skip_connections[i][1],
                out_features=skip_connections[i][1] * mult,
                features_height=skip_connections[i][2],
                features_width=skip_connections[i][3],
                edges=256,
                filters=128,
                apply_bias=True,
                trainable=True
            ) (skip_connections[i][0])

            skip_connections[i][0] = tf.keras.layers.ELU () (skip_connections[i][0])

        # Doing the first Deconvolution operation
        x = GatedDeConvolution (channels=channels * (2**(downsamples-1)), kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU') (x)
        x = tf.keras.layers.Concatenate () ([x, skip_connections[len(skip_connections)-1][0]])

        # Decoder for Refine Network
        current = len (skip_connections) - 2
        for i in range (downsamples-1, 0, -1) :
            dilation_rate = 1
            count = 2
            if i == downsamples - 1 :
                dilation_rate = 2
                count = 3

            for j in range (count) :
                x = GatedConvolution (channels=channels*(2**i), kernel_size=3, stride=1, dilation=dilation_rate, padding='same', activation='ELU') (x)

            x = GatedDeConvolution (channels=channels*(2**(i-1)), kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU') (x)
            if current != -1 :
                x = tf.keras.layers.Concatenate () ([x, skip_connections[current][0]])
                current -= 1

        x = GatedConvolution (channels=channels, kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU') (x)
        x = GatedConvolution (channels=channels, kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU') (x)
        refine_out = GatedConvolution (channels=3, kernel_size=3, stride=1, dilation=1, padding='same', activation=None) (x)

        return tf.keras.Model (inputs=[input_img, input_mask], outputs=[coarse_out, refine_out])

    def build_discriminator (self, image_height, image_width) :
        input_img = tf.keras.layers.Input (shape=[image_height, image_width, 3])
        input_mask = tf.keras.layers.Input (shape=[image_height, image_width, 1])

        x = tf.keras.layers.Concatenate () ([input_img, input_mask])

        channels = 64
        x = GatedConvolution (channels=channels, kernel_size=3, stride=1, dilation=1, padding='same', activation='LeakyReLU') (x)
        for i in range (1, 7) :
            mult = (2**i) if (2**i) < 8 else 8
            x = GatedConvolution (channels=channels * mult, kernel_size=3, stride=2,dilation=1, padding='same', activation='LeakyReLU') (x)

        return tf.keras.Model (inputs=[input_img, input_mask], outputs=x)