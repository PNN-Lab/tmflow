from sympy import MatrixSymbol, Matrix
import numpy as np

from tmflow.utils.tensor_ops import tensor_product


def expand_weights(initial_weights, order, output_dim):
    num_of_elements = np.zeros(order + 1, dtype=int)

    weights = []

    for i in range(order + 1):
        if i == 0:
            num_of_elements[i] = output_dim
        else:
            num_of_elements[i] = output_dim ** i
        weights.append(np.zeros((output_dim, num_of_elements[i])))

    supportVector = Matrix(MatrixSymbol("V", output_dim, 1))
    supportDeleteVector = supportVector.copy()
    support_array = [supportVector, supportVector]
    for i in range(order - 1):
        supportDeleteVector = tensor_product(supportDeleteVector, supportVector)
        support_array.append(supportDeleteVector)
    weights[0] = initial_weights[0]
    for k in range(1, order + 1):
        i = 0
        j = 0
        while i < support_array[k].shape[0]:
            k = i - 1
            flag = True
            while k > -1:
                if support_array[k][i] == support_array[k][k]:
                    flag = False
                    break
                k -= 1
            if flag:
                for d in range(output_dim):
                    weights[k][d, i] = initial_weights[k][d, j]
                j += 1
            i += 1

    return weights
