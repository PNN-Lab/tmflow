import unittest
import numpy as np
import sympy
from sympy.matrices import Matrix

from tmflow.map import construct_tm_matrices


def get_lotka_matrices():
    a = 1.5
    b = 1
    c = 3

    p11 = np.zeros((2, 2))
    p12 = np.zeros((2, 3))

    p11[0, 0] = a
    p11[1, 1] = -c
    p12[0, 1] = -b
    p12[1, 1] = b

    return p11, p12


def get_generalized_lotka_matrices():
    a = 2.9851
    b = 3
    c = 2

    p11 = np.zeros((3, 3))
    p12 = np.zeros((3, 6))
    p13 = np.zeros((3, 10))

    p11[0, 0] = 1
    p11[1, 1] = -1
    p11[2, 2] = -b
    p12[0, 0] = c
    p12[0, 1] = -1
    p12[1, 1] = 1
    p13[0, 2] = -a
    p13[2, 2] = a

    return p11, p12, p13


def get_robertson_matrices():
    a = 0.04
    b = 1e4
    c = 3e7

    p11 = np.zeros((3, 3))
    p12 = np.zeros((3, 6))

    p11[0, 0] = -a
    p11[1, 0] = a
    p12[0, 4] = b
    p12[1, 4] = -b
    p12[1, 3] = -c
    p12[2, 3] = c

    return p11, p12


def get_symbolic_lotka_matrices(symbolic_params):
    b_factor = 200
    a_factor = 300
    c_factor = 100
    a_energy = 2000
    b_energy = 378
    c_energy = 333

    p11 = sympy.zeros(2, 2)
    p12 = sympy.zeros(2, 3)

    if len(symbolic_params) != 1:
        raise Exception('wrong symbolical parameters count')

    T = symbolic_params[0]

    def arrenius_law_sym(T: sympy.Symbol, factor: float, energy: float):
        gas_constant = 8.314462
        return factor * sympy.exp(-energy / gas_constant / T)

    p11[0, 0] = arrenius_law_sym(T, a_factor, a_energy)
    p11[1, 1] = -arrenius_law_sym(T, c_factor, c_energy)
    p12[0, 1] = -arrenius_law_sym(T, b_factor, b_energy)
    p12[1, 1] = arrenius_law_sym(T, b_factor, b_energy)

    return p11, p12


def get_van_der_pol_approx_matrices():
    p11 = np.array([[4.09925688e-05, 9.99996394e-01],
                    [-1.00004690e+00, 1.00051369e+00]])

    p12 = np.array([[-3.46988607e-06, -1.45334833e-05, -6.11147369e-06],
                    [-1.81869333e-05, -7.76054624e-05, -3.59632744e-05]])

    p13 = np.array([[-2.90432033e-07, 2.33695855e-05, -3.64726368e-05, 1.97569005e-07],
                    [-1.07073040e-05, -1.00013086e+00, 1.69594456e-04, -1.31742648e-04]])

    return p11, p12, p13


lotka_res = [
    Matrix([
        [1.00150112556271, 0.0],
        [0.0, 0.997004495503373]]),
    Matrix([
        [0.0, -0.00100000037500004, 0.0],
        [0.0, 0.000997752622891932, 0.0]]),
    Matrix([
        [0.0, -5.00000093750123e-7, 4.99250936656893e-7, 0.0],
        [0.0, 4.9925065582869e-7, -4.98502341220916e-7, 0.0]]),
    Matrix([
        [0.0, -1.66729204105247e-10, 6.65792603427895e-10, -1.66167602881103e-10, 0.0],
        [0.0, 1.66541760216269e-10, -6.65043839531477e-10, 1.65980664371463e-10, 0.0]])
]

lotka_generalized_res = [
    Matrix([
        [1.00100050016671, 0.0, 0.0],
        [0.0, 0.999000499833375, 0.0],
        [0.0, 0.0, 0.997004495503373]]),
    Matrix([
        [0.00200300233458385, -0.00100050016670834, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.000999500166625008, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    Matrix([
        [4.00800833933668e-6, -3.50366887596701e-6, -0.00298510049751669, 5.00000041666668e-7, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0],
        [0.0, 1.50000012500001e-6, 0.0, -4.9966679163334e-7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.00297913626274451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    Matrix([
        [8.02002602336187e-9, -9.84884674995788e-9, -1.19463741816264e-5, 3.00183425032651e-9, 4.47516403297415e-6, 0.0,
         -1.66583374986567e-10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.50125062521104e-9, 0.0, -1.66633348331248e-9, -1.49105832028116e-6, 0.0, 1.66500091630755e-10, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.96224619707186e-6, 0.0, -2.97913601448318e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
]

robertson_res = [
    Matrix([
        [0.999960000799989, 0.0, 0.0],
        [3.99992000106666e-5, 1.0, 0.0],
        [0.0, 0.0, 1.0]]),
    Matrix([
        [0.0, 0.0, 0.000199994666746666, 0.0, 9.99980000266664, 0.0],
        [-1.59995200089599e-5, -1.19998400016, -0.000199994666746666, -30000.0, -9.99980000266664, 0.0],
        [1.59995200089599e-5, 1.19998400016, 0.0, 30000.0, 0.0, 0.0]]),
    Matrix([
        [1.27993813480622e-9, 0.000159994560104529, -3.99987200234666e-5, 7.99982000239997, -3.99992000095999,
         -0.000666646666986663, 149998.00002, -149998.00002, -49.9986666866664, 0.0],
        [7.6784000707994e-6, 0.959815045849473, 0.000159994240119455, 47991.4001857576, 15.9996000063999,
         0.000666646666986663, 899850001.99998, 449994.00006, 49.9986666866664, 0.0],
        [-7.67968000893419e-6, -0.959975040409576, -0.000119995520095989, -47999.40000576, -11.9996800054399, 0.0,
         -900000000.0, -299996.000039999, 0.0, 0.0]]),
    Matrix([
        [-8.04513697747224e-10, -0.000140793838040861, 1.27845583411499e-5, -10.5596720083796, 1.91762569872827,
         0.000319988266766057, -389992.320092809, 119872.843943945, 39.9988800181432, 0.0016666133342236,
         -5999940000.47999, 2997470055.23928, 1499970.00036, 166.661666746666, 0.0],
        [-3.72881232955039e-6, -0.652559060600399, -0.000108779655649962, -48942.4709640136, -16.3171041298935,
         -0.000879964269827454, -1799378412.44145, -899854.124231951, -109.996400069342, -0.0016666133342236,
         -26991000089999.2, -17997290056.9193, -3499920.00115998, -166.661666746666, 0.0],
        [3.72961684324814e-6, 0.652699854438441, 9.59950973088115e-5, 48953.0306360219, 14.3994784311652,
         0.000559976003061394, 1799768404.76155, 779981.280288008, 69.9975200511992, 0.0, 26997000029999.7,
         14999820001.68, 1999950.00079999, 0.0, 0.0]])
]

lotka_symbolic_res = [
    Matrix([
        [1.14402660558758, 0.0],
        [0.0, 0.916216541219891]]),
    Matrix([
        [0.0, -0.188273572300225, 0.0],
        [0.0, 0.168561888487359, 0.0]]),
    Matrix([
        [0.0, -0.0166782244654976, 0.0154921841841041, 0.0],
        [0.0, 0.0155056741021235, -0.0143833057368873, 0.0]]),
    Matrix([
        [0.0, -0.00100385285833425, 0.00369372725994221, -0.000849857241842379, 0.0],
        [0.0, 0.000950889865159694, -0.0034956548815872, 0.000803535461576525, 0.0]])
]

van_der_pol_res = [
    Matrix([
        [0.99999954080415, 0.00100049666966652],
        [-0.00100054720093353, 1.00100051401553]]),
    Matrix([
        [-3.47170060988998e-9, -1.45769203616439e-8, -6.1428771126255e-9],
        [-1.81554745123427e-8, -7.76579984320328e-8, -3.60530701181408e-8]]),
    Matrix([
        [-1.4070946927531e-10, -4.76979721342721e-7, -3.67347691129489e-8, 1.13592905257877e-10],
        [4.8970982347514e-7, -0.00100113090923582, -8.31396268320807e-7, -1.32255549785324e-7]]),
    Matrix([
        [4.7106853073497e-16, 7.29427440942541e-12, 6.15629284443933e-12, 7.7468725571616e-15, 9.13919140484765e-16],
        [9.07242159810065e-12, 8.11270338706741e-11, 6.87229135153141e-11, 6.20558410575865e-12, 1.54613306915131e-14]])
]


class TestWeightCalculation(unittest.TestCase):
    """
    Integration tests for weights calculation.
    Test are performed by running multiple test functions that generate weight matrices
    and subsequent comparison of these matrices to the precalculated ones
    """

    # The produced weight matrices are slightly different on different systems,
    # so we need to fuzzy compare them with the reference ones
    relative_tolerance = 0.00000001

    def test_lotka_second_order(self):
        res = self.get_result(get_lotka_matrices, 2)

        self.assertTrue(self.are_lists_of_matrices_equal(res, lotka_res))

    def test_generalized_lotka(self):
        res = self.get_result(get_generalized_lotka_matrices, 3)

        self.assertTrue(self.are_lists_of_matrices_equal(res, lotka_generalized_res))

    def test_robertson(self):
        res = self.get_result(get_robertson_matrices, 3)

        self.assertTrue(self.are_lists_of_matrices_equal(res, robertson_res))

    def test_van_der_pol(self):
        res = self.get_result(get_van_der_pol_approx_matrices, 2)
        print(res)

        self.assertTrue(self.are_lists_of_matrices_equal(res, van_der_pol_res))

    def test_symbolic_lotka(self):
        res = self.get_symbolic_result(get_symbolic_lotka_matrices, 2, [sympy.Symbol('T')], [300])

        self.assertTrue(self.are_lists_of_matrices_equal(res, lotka_symbolic_res))

    def get_result(self, matrix_getter, system_dimension):
        polynomial_system_ode = matrix_getter()
        order = 4
        tm_step = 0.001  # mapping step of TM

        return construct_tm_matrices(polynomial_system_ode, system_dimension, order, tm_step)

    def get_symbolic_result(self, matrix_getter, system_dimension, symbolic_params, symbolic_values):
        polynomial_system_ode = matrix_getter(symbolic_params)
        order = 4
        tm_step = 0.001  # mapping step of TM

        return construct_tm_matrices(polynomial_system_ode, system_dimension, order, tm_step, symbolic_params,
                                     symbolic_values)

    def are_lists_of_matrices_equal(self, a, b) -> bool:
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            matrixA = np.array(a[i]).astype(np.float64)
            matrixB = np.array(b[i]).astype(np.float64)
            if matrixA.shape != matrixB.shape:
                return False
            if not np.allclose(matrixA, matrixB, self.relative_tolerance):
                return False
        return True


if __name__ == '__main__':
    unittest.main()
