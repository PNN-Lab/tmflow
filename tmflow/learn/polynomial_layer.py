import tensorflow as tf

from tmflow.learn.base_polynomial_layer import BasePolynomialLayer


class PolynomialLayer(BasePolynomialLayer):
    def __init__(self, output_dim, order=1, initial_weights=None, use_bias=True, w_trainable_properties=None, **kwargs):
        super().__init__(output_dim, order, initial_weights, use_bias, w_trainable_properties, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.kernel_dims = [input_dim ** (n + 1) for n in range(self.order)]

        self._init_weights()

    def call(self, x):
        res = self.bias
        x_degree = tf.ones_like(x[:, 0:1])

        for i in range(self.order):
            x_degree = tf.einsum('bi,bj->bij', x, x_degree)
            x_degree = tf.reshape(x_degree, [-1, self.kernel_dims[i]])
            res = res + tf.matmul(x_degree, self.kernel_weights[i])

        return res
