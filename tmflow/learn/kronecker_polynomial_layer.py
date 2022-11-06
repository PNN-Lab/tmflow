import tensorflow as tf
from scipy.special import comb

from tmflow.learn.base_polynomial_layer import BasePolynomialLayer


class KroneckerPolynomialLayer(BasePolynomialLayer):
    def __init__(self, output_dim, order=1, initial_weights=None, use_bias=True, w_trainable_properties=None, **kwargs):
        super().__init__(output_dim, order, initial_weights, use_bias, w_trainable_properties, **kwargs)

        def f(x, y):
            return tf.vectorized_map(lambda xy: tf.tensordot(xy[0], xy[1], axes=0), elems=(x, y))

        self.compiled_tensordot = tf.function(f)
        self.symmetry_mask = []

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.kernel_dims = [comb(input_dim + k, k + 1, exact=True) for k in range(self.order)]

        self.symmetry_mask = [tf.constant(1, dtype=tf.int32, shape=(input_dim,))]
        for i in range(1, self.order):
            mask_tensor_degree = tf.tensordot(self.symmetry_mask[0], self.symmetry_mask[i - 1], axes=0)

            perm = [j for j in range(2, i + 1)] + [0, 1]
            mask_tensor_degree = tf.transpose(mask_tensor_degree, perm)
            mask_tensor_degree = tf.linalg.band_part(mask_tensor_degree, 0, -1)

            perm = [i - 1, i] + [j for j in range(0, i - 1)]
            mask_tensor_degree = tf.transpose(mask_tensor_degree, perm)

            self.symmetry_mask.append(mask_tensor_degree)

        self._init_weights()

    def call(self, x):
        res = self.bias
        x_tensor_degree = tf.ones_like(x[:, 0])

        for i in range(self.order):
            x_tensor_degree = self.compiled_tensordot(x, x_tensor_degree)

            x_kronecker_degree = tf.boolean_mask(x_tensor_degree, self.symmetry_mask[i], axis=1)
            x_kronecker_degree = tf.reshape(x_kronecker_degree, [-1, self.kernel_dims[i]])

            res = res + tf.matmul(x_kronecker_degree, self.kernel_weights[i])

        return res

    def get_config(self):
        config = super().get_config()
        return config
