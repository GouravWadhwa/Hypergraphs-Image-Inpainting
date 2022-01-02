import tensorflow as tf


class GatedConvolution(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size, stride=1, dilation=1, padding="same", activation="ELU"):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.activation = activation

    def __call__(self, input):
        # Apply convolution to the Input features
        x = tf.keras.layers.Conv2D(
            self.channels, self.kernel_size, self.stride, self.padding, dilation_rate=self.dilation, kernel_initializer=tf.keras.initializers.glorot_normal()
        )(input)

        # If we have final layer then we don't apply any activation
        if self.channels == 3 and self.activation is None:
            return x

        x, y = tf.split(x, 2, 3)

        if self.activation == "LeakyReLU":
            x = tf.keras.layers.LeakyReLU()(x)
        elif self.activation == "ReLU":
            x = tf.keras.layers.ReLU()(x)
        elif self.activation == "ELU":
            x = tf.keras.layers.ELU()(x)
        else:
            print("NO ACTIVATION!!!")

        # Gated Convolutiopn
        y = tf.nn.sigmoid(y)
        x = x * y

        return x


# Gated Deconvolution layer -> Upsampling + Gated Convolution


class GatedDeConvolution(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size, stride=1, dilation=1, padding="same", activation="ELU"):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.activation = activation

    def __call__(self, input):
        x = tf.keras.layers.UpSampling2D(size=2)(input)
        x = GatedConvolution(self.channels, self.kernel_size, self.stride, self.dilation, self.padding, self.activation)(x)

        return x
