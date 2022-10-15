import tensorflow as tf
import warnings


class BasePolynomialLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, order = 1, initial_weights = None, use_bias = True):
        super(BasePolynomialLayer, self).__init__()
        self.order = order
        self.output_dim = output_dim
        self.initial_weights = initial_weights
        self.use_bias = use_bias

        self.kernel_weights = list()
        self.kernel_dims = list()

    def _init_weights(self):
        if self.initial_weights and (self.order+1 != len(self.initial_weights)):
            warnings.warn('The required initial weights length does not correspond to the order')

        initial_values = []
        if self.initial_weights:
            transposed_weights = [weights.T for weights in self.initial_weights]
            bias_initial_value = transposed_weights[0]
            initial_values = transposed_weights[1:]
        else:
            bias_initial_value = tf.zeros_initializer()(shape=(1, self.output_dim))

            initial_values.append(tf.ones_initializer()(shape=(self.kernel_dims[0],self.output_dim)))
            for i in range(1, self.order):
                shape = (self.kernel_dims[i], self.output_dim)
                initial_values.append(tf.zeros_initializer()(shape = shape))

        self.bias = tf.Variable(initial_value=bias_initial_value, trainable=self.use_bias, dtype=tf.float32)

        for value in initial_values:
            var = tf.Variable(initial_value = value, trainable = True, dtype=tf.float32)
            self.kernel_weights.append(var)

        self._trainable_weights.extend(self.kernel_weights)
