import tensorflow as tf

class HypergraphConv (tf.keras.layers.Layer) :
    def __init__(
        self, 
        in_features,                                                                                              # Input Channels
        out_features,                                                                                             # Output Channels
        features_height,                                                                                          # Spatial height of features
        features_width,                                                                                           # Spatial width of features 
        edges,                                                                                                    # Number of edges in hypergraph convolution - A Hyperparamter
        filters=64,                                                                                               # Intermeditate channels for phi and lambda matrices - A Hyperparameter
        apply_bias=True,                                                                                          
        trainable=True, 
        name=None, 
        dtype=None, 
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.features_height = features_height
        self.features_width = features_width
        self.vertices = self.features_height * self.features_width
        self.edges = edges
        self.apply_bias = apply_bias
        self.trainable = trainable
        self.filters = filters

        # Make a weight of size (input channels * output channels) for applying the hypergraph convolution
        self.weight_2 = self.add_weight (
            name='Weight_2',
            shape=[self.in_features, self.out_features],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=self.trainable
        )

        # If applying bias on the output features, make a weight of size (output channels) 
        if apply_bias :
            self.bias_2 = self.add_weight (
                name='Bias_2',
                shape=[self.out_features],
                dtype=tf.float32,
                initializer=tf.keras.initializers.glorot_normal(),
                trainable=self.trainable
            )

    def call (self, x) :
        # Summary of the hypergraph convolution
        # x shape - self.features_height * self.features_width * self.in_features
        # features - x
        # H = phi * A * phi.T * M
        # phi = conv2D (features)
        # A = tf.linalg.tensor_diag (conv2D (gloabalAveragePooling (features))
        # D = tf.linalg.tensor_diag (tf.math.reduce_sum (H, axis=1))
        # B = tf.linalg.tensor_diag (tf.math.reduce_sum (H, axis=0))
        # L = I - D^(-0.5) H B^(-1) H.T D^(-0.5)
        # out = L * features * self.weight_2 + self.bias_2

        # Phi Matrix
        phi = tf.keras.layers.Conv2D (self.filters, kernel_size=1, strides=1, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()) (x)
        phi = tf.reshape (phi, shape=(-1, self.vertices, self.filters))
        
        # Lambda Matrix
        A = tf.keras.layers.GlobalAveragePooling2D () (x)
        A = tf.expand_dims (tf.expand_dims (A, axis=1), axis=1)
        A = tf.keras.layers.Conv2D (self.filters, kernel_size=1, strides=1, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()) (A)
        A = tf.linalg.diag (tf.squeeze (A))

        # Omega Matrix
        M = tf.keras.layers.Conv2D (self.edges, kernel_size=7, strides=1, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal ()) (x)
        M = tf.reshape (M, shape=(-1, self.vertices, self.edges))
        
        # Incidence matrix
        # H = | phi * lambda * phi.T * omega |
        H = tf.matmul (phi, tf.matmul (A, tf.matmul (tf.transpose (phi, perm=[0, 2, 1]), M)))
        H = tf.math.abs (H)
        
        # Degree matrix
        D = tf.math.reduce_sum (H, axis=2)

        # Mutlpying with the incidence matrix to ensure no matrix developed is of large size - (number of vertices * number of vertices)
        D_H = tf.multiply (tf.expand_dims (tf.math.pow (D, -0.5), axis=-1), H)
        
        # Edge degree Matrix
        B = tf.math.reduce_sum (H, axis=1)
        B = tf.linalg.diag (tf.math.pow (B, -1))
        
        # Reshape the input features to apply the Hypergraph Convolution
        features = tf.reshape (x, shape=(-1, self.vertices, self.in_features))
        
        # Hypergraph Convolution
        out = features - tf.matmul (D_H, tf.matmul (B, tf.matmul (tf.transpose (D_H, perm=[0, 2, 1]), features)))
        out = tf.matmul (out, self.weight_2)
        
        if self.apply_bias :
            out = out + self.bias_2

        # Reshape to output size
        out = tf.reshape (out, shape=(-1, self.features_height, self.features_width, self.out_features))
        
        return out