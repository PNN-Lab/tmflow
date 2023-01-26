from sympy import Symbol
import numpy as np
import sympy
import math


class OdeRobertson:
    def __init__(self, ndim, a, b, c) -> None:
        self.ndim = ndim
        self.a = a
        self.b = b
        self.c = c

    def f(self, x, t):
        f = np.zeros(self.ndim)
        f[0] = -self.a * x[0] + self.b * x[1] * x[2]
        f[1] = self.a * x[0] - self.b * x[1] * x[2] - self.c * x[1] * x[1]
        f[2] = self.c * x[1] * x[1]
        return f

    def get_rhs_matrices(self):
        p11 = np.zeros((3, 3))
        p12 = np.zeros((3, 6))

        p11[0, 0] = -self.a
        p11[1, 0] = self.a
        p12[0, 4] = self.b
        p12[1, 4] = -self.b
        p12[1, 3] = -self.c
        p12[2, 3] = self.c

        return p11, p12


class OdeLotka:
    def __init__(self, ndim, a, b, c) -> None:
        self.ndim = ndim
        self.a = a
        self.b = b
        self.c = c

    def f(self, x, t):
        f = np.zeros(self.ndim)
        f[0] = self.a * x[0] - self.b * x[0] * x[1]
        f[1] = self.b * x[0] * x[1] - self.c * x[1]
        return f

    def get_rhs_matrices(self):
        p11 = np.zeros((2, 2))
        p12 = np.zeros((2, 3))

        p11[0, 0] = self.a
        p11[1, 1] = -self.c
        p12[0, 1] = -self.b
        p12[1, 1] = self.b

        return p11, p12


class OdeLotkaGeneralized:
    def __init__(self, ndim, a, b, c) -> None:
        self.ndim = ndim
        self.a = a
        self.b = b
        self.c = c

    def f(self, x, t, a=2.9851, b=3, c=2):
        f = [0, 0, 0]

        f[0] = x[0] - x[0] * x[1] + c * x[0] * x[0] - a * x[0] * x[0] * x[2]
        f[1] = -x[1] + x[0] * x[1]
        f[2] = (-b) * x[2] + a * x[0] * x[0] * x[2]
        return f

    def get_rhs_matrices(self):
        p11 = np.zeros((3, 3))
        p12 = np.zeros((3, 6))
        p13 = np.zeros((3, 10))

        p11[0, 0] = 1
        p11[1, 1] = -1
        p11[2, 2] = -self.b

        p12[0, 0] = self.c
        p12[0, 1] = -1
        p12[1, 1] = 1

        p13[0, 2] = -self.a
        p13[2, 2] = self.a

        return p11, p12, p13


class OdeLotkaSymbolic:
    def __init__(self, ndim, a_factor, b_factor, c_factor, a_energy, b_energy, c_energy) -> None:
        self.ndim = ndim
        self.a_factor = a_factor
        self.b_factor = b_factor
        self.c_factor = c_factor
        self.a_energy = a_energy
        self.b_energy = b_energy
        self.c_energy = c_energy

    def arrenius_law_sym(self, T: Symbol, factor: float, energy: float):
        gas_constant = 8.314462
        return factor * sympy.exp(-energy / gas_constant / T)

    def arrenius_law(self, T: float, factor: float, energy: float):
        gas_constant = 8.314462
        return factor * math.exp(-energy / gas_constant / T)

    def f(self, x, t, T):
        f = np.zeros(self.ndim)
        f[0] = self.arrenius_law(T, self.a_factor, self.a_energy) * x[0] - self.arrenius_law(T, self.b_factor, self.b_energy) * x[0] * x[1]
        f[1] = self.arrenius_law(T, self.b_factor, self.b_energy) * x[0] * x[1] - self.arrenius_law(T, self.c_factor, self.c_energy) * x[1]
        return f

    def get_rhs_matrices(self, symb_params):
        p11 = sympy.zeros(2, 2)
        p12 = sympy.zeros(2, 3)

        T = symb_params[0]

        p11[0, 0] = self.arrenius_law_sym(T, self.a_factor, self.a_energy)
        p11[1, 1] = -self.arrenius_law_sym(T, self.c_factor, self.c_energy)
        p12[0, 1] = -self.arrenius_law_sym(T, self.b_factor, self.b_energy)
        p12[1, 1] = self.arrenius_law_sym(T, self.b_factor, self.b_energy)

        return p11, p12
