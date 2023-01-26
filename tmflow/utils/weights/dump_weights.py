import pickle
import numpy as np


def dump_pnn_weight_matrices(order, input_matrices, filename):
    """
    Dump weights that were obtained from a PNN in pickle format as a single matrix
    """
    weights = [matrix.T for matrix in input_matrices]
    bias_vector = weights[-1].reshape((-1, 1))
    weight_matrix = combine_weight_matrices(order, input_matrices, bias_vector)
    with open(f'{filename}', 'wb') as fid:
        pickle.dump(weight_matrix, fid)


def combine_weight_matrices(order, input_matrices, bias_vector=None):
    # start combining from the bias vector if it is present, otherwise fill the bias with zeros
    if not bias_vector:
        ndim = input_matrices[0].shape[0]
        bias_vector = np.zeros((ndim, 1))

    weights = bias_vector
    for i in range(0, order):
        weights = np.append(weights, np.fliplr(input_matrices[i]), axis=1)
    return weights
