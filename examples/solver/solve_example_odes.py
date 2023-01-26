import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from sympy import Symbol

from tmflow.utils.solver import TMSolver, run_tm_solver

from example_odes import OdeLotka, OdeRobertson, OdeLotkaSymbolic, OdeLotkaGeneralized


def solve_lotka_ode():
    equation = OdeLotka(ndim=2, a=1.5, b=1, c=3)
    xy0 = [0.5, 2.5]

    t0 = 0
    dt = 0.001
    end_time = t0 + 50
    n_step = (int)((end_time - t0) // dt)
    order = 3

    tm_sol = np.zeros((equation.ndim, n_step + 1))
    tm_sol[:, 0] = xy0
    t = [t0 + i * dt for i in range(0, n_step + 1)]

    solver = TMSolver(order=order, ode_dim=equation.ndim, ode_rhs=equation.get_rhs_matrices(), map_step=dt, calc_weights=True)

    tm_sol = run_tm_solver(xy0, t, solver)

    num_sol = odeint(equation.f, xy0, t)

    plot_solutions(equation.ndim, tm_sol, num_sol, t, solver.order, solver.map_step)


def solve_robertson_ode():
    equation = OdeRobertson(ndim=3, a=0.04, b=1e4, c=3e7)
    xy0 = [1, 0, 0]
    # xy0 = [0.6736042454040031, 7.70902978789662e-06, 0.32638804556620876]

    t0 = 0
    dt = 0.0001
    end_time = t0 + 50
    n_step = (int)((end_time - t0) / dt)
    order = 4

    tm_sol = np.zeros((equation.ndim, n_step + 1))
    tm_sol[:, 0] = xy0
    t = [t0 + i * dt for i in range(0, n_step + 1)]

    solver = TMSolver(order=order, ode_dim=equation.ndim, ode_rhs=equation.get_rhs_matrices(), map_step=dt, calc_weights=True)

    tm_sol = run_tm_solver(xy0, t, solver)

    f = lambda t, x: equation.f(x, t)
    num_sol = solve_ivp(f, t_span=[t0, end_time], y0=xy0, t_eval=t, method='Radau', rtol=1e-12).y.T

    plot_solutions(equation.ndim, tm_sol, num_sol, t, solver.order, solver.map_step)


def solve_symbolic_lotka_ode():
    equation = OdeLotkaSymbolic(ndim=2, a_factor=300, b_factor=200, c_factor=100, a_energy=2000, b_energy=378, c_energy=333)
    xy0 = [0.5, 2.5]
    t0 = 0
    dt = 0.5e-4
    end_time = t0 + 2

    n_step = (int)((end_time - t0) / dt)

    order = 3
    T = Symbol('T')

    tm_sol = np.zeros((equation.ndim, n_step + 1))
    tm_sol[:, 0] = xy0
    t = [t0 + i * dt for i in range(0, n_step + 1)]

    fixed_temperature_point = 300

    solver = TMSolver(order=order, ode_dim=equation.ndim, ode_rhs=equation.get_rhs_matrices([T]), map_step=dt, calc_weights=True, symbolic_args=[T], numerical_args=[fixed_temperature_point])

    tm_sol = run_tm_solver(xy0, t, solver)

    num_sol = odeint(equation.f, xy0, t, args=tuple([fixed_temperature_point]))

    plot_solutions(equation.ndim, tm_sol, num_sol, t, solver.order, solver.map_step)


def solve_generalized_lotka_ode():
    equation = OdeLotkaGeneralized(ndim=3, a=2.9851, b=3, c=2)
    xy0 = [16, 24, 18]
    t0 = 0

    dt = 0.5e-4
    end_time = t0 + 2

    n_step = (int)((end_time - t0) / dt)

    order = 4

    tm_sol = np.zeros((equation.ndim, n_step + 1))
    tm_sol[:, 0] = xy0
    t = [t0 + i * dt for i in range(0, n_step + 1)]

    solver = TMSolver(order=order, ode_dim=equation.ndim, ode_rhs=equation.get_rhs_matrices(), map_step=dt, calc_weights=True)

    tm_sol = run_tm_solver(xy0, t, solver)

    num_sol = odeint(equation.f, xy0, t)
    plot_solutions(equation.ndim, tm_sol, num_sol, t, solver.order, solver.map_step)


def plot_solutions(dim, tm_sol, num_sol, time_mas, order, map_step):
    fig, ax = plt.subplots(2, 1)
    line_symbols = [':', '-', '-.']
    ax[0].set_title(f'TM order {order}, time step {map_step}')
    for i in range(dim):
        ax[0].plot(time_mas, num_sol[:, i], f'{line_symbols[i]}r', label=f'x{i + 1} lsoda')
        ax[0].plot(time_mas, tm_sol[:, i], f'{line_symbols[i]}b', label=f'x{i + 1} TM')

    ax[0].legend(loc="upper right")

    # errors
    for i in range(dim):
        ax[1].plot(time_mas, np.abs(tm_sol[:, i] - num_sol[:, i]), label=f'x{i + 1} abs error')

    ax[1].legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    solve_lotka_ode()
    solve_robertson_ode()
    solve_generalized_lotka_ode()
    solve_symbolic_lotka_ode()
