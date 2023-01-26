from sympy.matrices import Matrix
from typing import Union, List
import numpy.typing as npt


def tensor_product(A: Union[Matrix, npt.ArrayLike], B: Union[Matrix, npt.ArrayLike]) -> Matrix:
    """
    Tensor product of two matrices.

    Computes the tensor product `C` of two matrices `A` and `B` given either in symbolic
    or numerical form :math:`C=A\otimes B`

    :param A: the left operand of the tensor product, shape(n,m)
    :param B: the right operand of the tensor product, shape(k,l)

    :return: The tensor product in a symbolic form, shape(n*k+k,m*l+l)
    """

    n_row_A = A.shape[0]
    n_col_A = A.shape[1]
    n_row_B = B.shape[0]
    n_col_B = B.shape[1]
    n_row_C = n_row_A * n_row_B
    n_col_C = n_col_A * n_col_B
    C = Matrix.zeros(n_row_C, n_col_C)
    for index_r_A in range(n_row_A):
        for index_c_A in range(n_col_A):
            for index_r_B in range(n_row_B):
                for index_c_B in range(n_col_B):
                    C[index_r_A * n_row_B + index_r_B, index_c_A * n_col_B + index_c_B] = \
                        A[index_r_A, index_c_A] * B[index_r_B, index_c_B]
    return C


def maps_tensor_product(A: List[Matrix], B: List[Matrix]) -> List[Matrix]:
    """
    Tensor product of two matrix maps.

    Computes the tensor product `C` of two matrix maps `A` and `B`.
    Accepts matrix maps as a list of matrices in symbolic or numerical form:

    :math:`A\cdot X_d={R_1, R_2, R_3,....}\cdot {X, X^{[2]}, X^{[3]},...}^T
    = R_1\cdot X + R_2 \cdot X^{[2]} + R_3\cdot X^{[3]}+...`

    :math:`B\cdot X_d={Q_1, Q_2, Q_3,....}\cdot {X, X^{[2]}, X^{[3]},...}^T
    = Q_1\cdot X + Q_2 \cdot X^{[2]} + Q_3\cdot X^{[3]}+...`

    Returns the new map as a list of ``Matrix`` symbolic objects

    :param A: the left operand of the map tensor product, a list of Matrix objects
    :param B: the right operand of the map tensor product, a list of Matrix objects

    :return: the list of matrix maps tensor product in a symbolic form
    """

    n_col_A = len(A)
    n_col_B = len(B)
    C = []
    for index_c_A in range(n_col_A):
        for index_c_B in range(n_col_B):
            C.append(tensor_product(A[index_c_A], B[index_c_B]))
    return C
