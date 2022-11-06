import pickle
import math
import numpy as np
from typing import List, Union


def convert_initial_weights(dim: int, order: int, filename: str,
                            normalization_factor: Union[float, None] = None) -> List[np.ndarray]:
    """
    Converts pickled initial weights provided by the tmflow.map module that were stored as a single matrix
    with dimension :math:`[dim, \sum_\limits_{k=0}^{order} C^{dim+k-1}_k]`
    to format suitable for PNN initialization
    :param dim: system dimension
    :param order: order of non-linearity
    :param filename: filename of pickled weights
    :param normalization_factor: value that is used to normalize weights (usually, max possible system variables' value)
    :return: initial matrices for PNN (list of matrices with length equals order+1)
    """

    init_matrices = [np.zeros(dim)]

    with open(filename, 'rb') as input_file:
        big_matrix = pickle.load(input_file)

    pre_idx = 1
    for i in range(1, order + 1):

        next_idx = int(math.factorial(dim + i - 1) // (math.factorial(i) * math.factorial(dim - 1))) + pre_idx

        init_matrices.append(np.fliplr(np.array(big_matrix[:, pre_idx:next_idx])))
        pre_idx += int(math.factorial(dim + i - 1) // (math.factorial(i) * math.factorial(dim - 1)))

        if normalization_factor:
            init_matrices[i - 1] = init_matrices[i - 1] * normalization_factor ** (i - 2)

    return init_matrices
