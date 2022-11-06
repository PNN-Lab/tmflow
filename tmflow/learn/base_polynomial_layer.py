import tensorflow as tf
import warnings


class BasePolynomialLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, order=1, initial_weights=None, use_bias=True, w_trainable_properties=None, **kwargs):
        super(BasePolynomialLayer, self).__init__(**kwargs)
        self.order = order
        self.output_dim = output_dim
        self.initial_weights = initial_weights
        self.use_bias = use_bias

        if w_trainable_properties:
            assert (order == len(w_trainable_properties))
            self.w_trainable_properties = w_trainable_properties
        else:
            self.w_trainable_properties = [True] * order

        self.kernel_weights = list()
        self.kernel_dims = list()

    def _init_weights(self):
        not_fully_initialized = False
        if self.initial_weights:
            if self.order + 1 < len(self.initial_weights):
                warnings.warn('The initial weights length is larger than the order, extra weights will be trimmed')
            elif self.order + 1 > len(self.initial_weights):
                not_fully_initialized = True
                warnings.warn('The initial weights length is less than the order, the rest of weights will be initialized by zeros')

        initial_values = []
        if self.initial_weights:
            transposed_weights = [weights.T for weights in self.initial_weights]
            bias_initial_value = transposed_weights[0]
            initial_values = transposed_weights[1:]
        else:
            bias_initial_value = tf.zeros_initializer()(shape=(1, self.output_dim))

            initial_values.append(tf.eye(self.kernel_dims[0], self.output_dim))
            for i in range(1, self.order):
                shape = (self.kernel_dims[i], self.output_dim)
                initial_values.append(tf.zeros_initializer()(shape=shape))

        if not_fully_initialized:
            shape = (self.kernel_dims[self.order-1], self.output_dim)
            initial_values.append(tf.zeros_initializer()(shape=shape))

        self.bias = tf.Variable(initial_value=bias_initial_value, trainable=self.use_bias, dtype=tf.float32)

        for i, value in enumerate(initial_values):
            var = tf.Variable(initial_value=value, trainable=self.w_trainable_properties[i], dtype=tf.float32,
                              name=f'W_{i + 1}')

            self.kernel_weights.append(var)

        trainable_weights = [kernel for kernel in self.kernel_weights if kernel.trainable]
        self._trainable_weights.extend(trainable_weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            "order": self.order,
            "output_dim": self.output_dim,
            "initial_weights": self.initial_weights,
            "use_bias": self.use_bias,
            "w_trainable_properties": self.w_trainable_properties,
        })
        return config
