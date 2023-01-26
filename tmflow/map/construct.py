from sympy import MatrixSymbol
from sympy.matrices import Matrix
from sympy.utilities.lambdify import lambdify

import math
import numpy as np
from scipy.integrate import solve_ivp

from typing import Union, List, Tuple, Callable
import numpy.typing as npt

from tmflow.utils.tensor_ops import tensor_product


def __truncated_maps_tensor_product(A: Union[List[Matrix], List[npt.ArrayLike]],
                                    B: Union[List[Matrix], List[npt.ArrayLike]],
                                    sub_A_orders: List[npt.ArrayLike],
                                    sub_B_orders: List[npt.ArrayLike],
                                    max_order: int) -> List[Matrix]:
    """
    Tensor product of two matrix maps truncated till the required order of nonlinearity.

    Computes the tensor product `C` of two matrix maps `A` and `B`, neglecting terms of order higher than `max_order`.
    Accepts matrix maps as a list of matrices. The sequence of matrices in the list corresponds to
    the orders of nonlinearities given in subsidiary arrays `sub_A_orders` and `sub_B_orders`
    Returns the list of ``Matrix`` objects

    :param A: the left operand of the map tensor product, a list of
        Matrix, if `A[i]` in symbolic form, or
        a list of array_like, if `A[i]` in numerical form
    :param B: the right operand of the map tensor product, a list of
        Matrix if `B[i]` in symbolic form, or
        a list of array_like if `B[i]` in numerical form
    :sub_A_orders: subsidiary array contains sequent orders of matrices in map `A`
    :sub_B_orders: subsidiary array contains sequent orders of matrices in map `B`
    :max_order: limits the order of nonlinearity in the resulting tensor product
    :return: the tensor product of matrix maps till `max_order` of nonlinearity returned in a symbolic form
    """

    n_col_A = len(A)
    n_col_B = len(B)
    C = []
    for index_c_A in range(n_col_A):
        for index_c_B in range(n_col_B):
            if sub_A_orders[index_c_A] + sub_B_orders[index_c_B] <= max_order:
                C.append(tensor_product(A[index_c_A], B[index_c_B]))

    return C


def __reduce_map_tensor_product(A: List[Matrix], X_d: List[Matrix]) -> Tuple[List[Matrix], List[Matrix]]:
    """
    Reduces duplicate elements in the tensor product result of two matrix maps.

    The resulting matrix map of the tensor product is given as a list `A` of matrices `{A_1, A_2, A_3,....}`
    and list `X_d` of vectors `X_i` corresponding to `A_i`:

    :math:`M = A\cdot X_d^T={A_1, A_2, A_3,....}\cdot {X_1, X_2, X_3,...}^T
    = A_1\cdot X + A_2 \cdot X^{[2]} + A_3\cdot X^{[3]}+...`

    Where duplicate matrices in `A` are summarized, and identical vectors in `X_d` are deleted

    :param A: matrices of the tensor product map
    :param X_d: corresponding degrees of phase vector
    :return: reduced matrix map (`A`, `X_d`)
    """

    n_col = len(X_d)
    i = 0
    while i < n_col:
        j = i + 1
        while j < n_col:
            if A[i].shape == A[j].shape:
                A[i] += A[j]
                del A[j]
                del X_d[j]
                n_col -= 1
            else:
                j += 1
        i += 1
    return A, X_d


def __delete_rows_tensor_product(A_array: Union[List[Matrix], List[npt.ArrayLike]], x: Matrix) -> List[Matrix]:
    """
    Deletes duplicate rows in the list of tensor products of two matrices.

    Duplicate rows are found via accordance of matrix `A` rows and elements of the auxiliary phase vector `x`.

    :param A_array: matrix - the result of the tensor product of two matrices
    :param x: auxiliary phase vector
    :return: matrix A with removed duplicate rows
    """

    A_array = A_array.copy()
    n_row_A = A_array[0].shape[0]
    i = 0
    x = x.copy()
    while i < n_row_A:
        j = i + 1
        while j < n_row_A:
            if str(x[i]) == str(x[j]):
                for a in A_array:
                    a.row_del(j)
                x.row_del(j)
                n_row_A -= 1
            else:
                j += 1
        i += 1
    return A_array


def __reduce_columns_tensor_product(A_array: Union[List[Matrix], List[npt.ArrayLike]],
                                    X_array: List[Matrix]) -> List[Matrix]:
    """
    Reduces duplicate columns in the list of results of tensor product of two matrices.

    Duplicate columns are found via accordance of matrices' columns in `A_array`
    and vectors' rows in the auxiliary `X_array`.
    Duplicated columns in matrices of `A_array` are summarized, and duplicated rows in vectors of `X_array` are deleted.

    :param A_array: list of matrices of tensor product block matrix (array)
    :param X_array: list of vectors block vector (array)
    :return: list of reduced matrices
    """

    X = X_array.copy()
    A = A_array.copy()
    for idx, x in enumerate(X):
        n_row_A = x.shape[0]
        i = 0
        while i < n_row_A:
            j = i + 1
            while j < n_row_A:
                if x[i] == x[j]:
                    A[idx][:, i] += A[idx][:, j]
                    A[idx].col_del(j)
                    x.row_del(j)
                    n_row_A -= 1
                else:
                    j += 1
            i += 1
    return A


def __delete_rows_vector(X: Matrix) -> Matrix:
    """
    Deletes duplicate rows of a symbolic vector.

    :param X: vector
    :return: vector `X` with removed duplicate rows
    """

    n_row_x = X.shape[0]
    i = 0
    while i < n_row_x:
        j = i + 1
        while j < n_row_x:
            if X[i] == X[j]:
                X.row_del(j)
                n_row_x -= 1
            else:
                j += 1
        i += 1
    return X


def __next_sup_matrix(s_m: npt.ArrayLike, b_s_m: npt.ArrayLike, max_order: int) -> npt.ArrayLike:
    """
    Calculates the next auxiliary vector storing information about the sequence of degrees in the list of matrices

    :param s_m: auxiliary vector
    :param b_s_m: base auxiliary vector (vector of 1 degree)
    :param max_order: the maximum order of the system
    :return: new helper vector
    """

    sup_matrix_out = []
    for i in s_m:
        for j in b_s_m:
            if i + j <= max_order:
                sup_matrix_out.append(i + j)
    return sup_matrix_out


def __mapping_lambdify(order: int, R_rhs: Union[List[Matrix], List[npt.ArrayLike]],
                       R_symb: List[Matrix], sym_params=None) -> Tuple[Callable, int]:
    """
    Makes callable object from a symbolic expression of tm_ode r.h.s.

    :param order: order of nonlinearity
    :param R_rhs: list of symbolic expressions of tm_ode r.h.s
    :param R_symb: list of symbolic variables corresponding to R_i in tm_ode
    :param sym_params: (optional) additional symbolic parameters in the tm_ode r.h.s. that are substituted later
    :return: lambda function and total number of variable in all matrices
    """

    _W = []
    for i in range(order):
        w = R_rhs[i]
        for j in w:
            _W.append(j)
    _W = Matrix(_W)
    R_coeff = []
    for r in R_symb:
        for coeff in r:
            R_coeff.append(coeff)

    if sym_params:
        R_coeff.extend(sym_params)

    return lambdify(R_coeff, _W, 'numpy'), len(_W)


def __generate_tm_ode(dim: int, order: int, P_matrix_array: Union[Tuple[Matrix], Tuple[npt.ArrayLike]],
                      sym_params=None) -> Tuple[Callable, int]:
    """
    Generates function of tm_ode r.h.s.

    :param dim: system dimension
    :param order: required order of nonlinearity of TM
    :param P_matrix_array: array of matrices P_i from original ODE polynomial r.h.s.
    :param sym_params: possible symbolic parameters in the original ODE system (to be substituted later default = None)
    :return: and total number of variable in all matrices
    """
    R = [Matrix(MatrixSymbol('R' + str(i), dim,
                             int(math.factorial(dim + i-1)//(math.factorial(i)*math.factorial(dim-1)))))
         for i in range(1, order+1)]
    X = Matrix(MatrixSymbol('X', dim, 1))
    sup_base_matrix = [i for i in range(1, order + 1, 1)]
    sup_matrix = sup_base_matrix.copy()
    R_array = [r for r in R]
    X_array = [X]
    cron_X = X
    for i in range(1, order):
        cron_X = tensor_product(__delete_rows_vector(cron_X), X)
        X_array.append(cron_X)

    before_del_cron_X = X_array
    cron_R = R_array
    before_del_cron_R = R_array
    W = [P_matrix_array[0] * r for r in cron_R]

    assert dim > 1, "Only an ODE system with 2 or more equations is supported at the moment"
    for idx, p in enumerate(P_matrix_array):
        R_line = [p * r for r in cron_R]
        if idx != 0:
            for idw, w in enumerate(W):
                for r in R_line:
                    if w.shape == r.shape:
                        W[idw] += r

        if idx + 1 != len(P_matrix_array):
            before_del_cron_X = __truncated_maps_tensor_product(before_del_cron_X,
                                                                X_array,
                                                                sub_A_orders=sup_matrix,
                                                                sub_B_orders=sup_base_matrix,
                                                                max_order=order)

            before_del_cron_R = __truncated_maps_tensor_product(before_del_cron_R,
                                                                R_array,
                                                                sub_A_orders=sup_matrix,
                                                                sub_B_orders=sup_base_matrix,
                                                                max_order=order)

            sup_matrix = __next_sup_matrix(s_m=sup_matrix,
                                           b_s_m=sup_base_matrix,
                                           max_order=order)

            cron_X = [x.copy() for x in before_del_cron_X]
            cron_R = [r.copy() for r in before_del_cron_R]
            cron_R = __reduce_columns_tensor_product(cron_R, cron_X)
            cron_R, cron_X = __reduce_map_tensor_product(cron_R, cron_X)
            cron_R = __delete_rows_tensor_product(cron_R, before_del_cron_X[0])

    return __mapping_lambdify(order, W, R, sym_params)


def __solve_tm_ode(x0: float, x1: float, variable_count: int, dim: int, tm_ode_rhs: any, symbolic_values=None) -> npt.ArrayLike:
    """
    Integrates tm_ode at time interval `[x0,x1]`

    :param x0: start point
    :param x1: end point
    :param variable_count: total variable count in all matrices
    :param dim: system dimension
    :param tm_ode_rhs: function of tm_ode r.h.s.
    :param symbolic_values: numerical values of symbolic parameters if the tm_ode r.h.s. was created with symbolic parameters
    :return: TM matrices
    """
    def pend(t, y, *num_args):
        args = tuple(y) + num_args
        return tm_ode_rhs(*args).reshape(1, variable_count)[0]

    Y0 = [0.0 for w in range(variable_count)]
    for i in range(dim):
        Y0[i * dim + i] = 1
    y0 = np.array(Y0)
    arguments = tuple(symbolic_values) if symbolic_values else None
    res = solve_ivp(pend, (x0, x1), y0, t_eval=[x1], args=arguments)
    return res.y[:, -1]


def construct_tm_matrices(system_matrices: Union[Tuple[npt.ArrayLike], Tuple[Matrix]],
                          system_dim: int, tm_order: int, tm_step: float,
                          sym_params=None, num_params=None) -> npt.ArrayLike:
    """
    Calculates matrices of Taylor mapping for an autonomous polynomial system of ordinary differential equations
    :math:`\frac{d\mathbf{X}}{dt} = P_0+P_1\mathbf{X}+P_2\mathbf{X}^{[2]}+P_3\mathbf{X}^{[3]}+...`

    :param system_matrices: tuple of NumPy arrays or symbolic matrices with parameters :math:`(P_0,P_1,P_2,P_3,...)`
    :param system_dim: dimension of vector :math:`\mathbf{X}`
    :param tm_order: requested number of nonlinear terms in Taylor mapping
    :param tm_step: Taylor mapping step
    :param sym_params: (optional) symbolic parameters for the ODE system
    :param num_params: numerical values of symbolic parameters that were passed with `sym_params`
    :return: array of Taylor mapping matrices for the required order of nonlinearity (`tm_order`)
    and mapping step (`tm_step`)
    """
    matrices, variable_count = __generate_tm_ode(system_dim, tm_order, system_matrices, sym_params)
    vector_of_coeffs = __solve_tm_ode(0, tm_step, variable_count, system_dim, matrices, num_params)
    return __convert_vector_to_matrix(vector_of_coeffs, tm_order, system_dim)


def __convert_vector_to_matrix(vector_coefficients: npt.ArrayLike, tm_order: int, system_dim: int) -> npt.ArrayLike:
    """
    Converts vector of tm coefficients to list of matrices of the corresponding dimension

    :param vector_coefficients: vector of tm coefficients
    :param tm_order: requested number of nonlinear terms in Taylor mapping
    :param system_dim: dimension of vector :math:`\mathbf{X}`
    """
    matrices_r = []
    a = 0
    for i in range(tm_order):
        m = []
        column_count = int(math.factorial(system_dim + i) //
                           (math.factorial(i + 1) * math.factorial(system_dim - 1)))
        for p in range(system_dim):
            row = []
            for j in range(column_count):
                row.append(vector_coefficients[a])
                a += 1
            m.append(row)
        matrices_r.append(Matrix(m))
    return matrices_r
