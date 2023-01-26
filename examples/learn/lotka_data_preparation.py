import pandas as pd
import numpy as np
import numpy.typing as npt
import itertools
from scipy.stats import uniform
from scipy.integrate import solve_ivp


def init_lotka_equation(a: float, b: float, c: float):
    dim = 2

    def f(t: float, x: npt.ArrayLike):
        y = np.zeros(dim)
        y[0] = a * x[0] - b * x[0] * x[1]
        y[1] = b * x[0] * x[1] - c * x[1]
        return y

    return f


def run_solve_ode(callable_equation, init_vals, time_points, method):
    return [solve_ivp(callable_equation,
                      t_span=[time_points[0], time_points[-1]],
                      y0=ini_data,
                      method=method,
                      t_eval=time_points).y.T
            for ini_data in init_vals]


def generate_uniform_points(lower_bound: float, upper_bound: float, length: int):
    random_generator = uniform(loc=lower_bound, scale=upper_bound - lower_bound)
    return random_generator.rvs(length)


def generate_training_data(callable_equation, trajectory_len, dt):
    time_points = [i * dt for i in range(0, trajectory_len)]

    # generate points for x1 and x2
    x1_init_vals = generate_uniform_points(lower_bound=1, upper_bound=15, length=20)
    x2_init_vals = generate_uniform_points(lower_bound=1, upper_bound=15, length=20)

    # make all kinds of combinations
    init_vals = list(itertools.product(x1_init_vals, x2_init_vals))
    total_num = len(init_vals)

    # create trajectories for initial data
    numerical_solution = run_solve_ode(callable_equation, init_vals, time_points, 'LSODA')
    solutions = np.concatenate(numerical_solution, axis=0)

    # create dataframe
    return pd.DataFrame(data=np.hstack((np.tile(time_points, total_num).reshape((-1, 1)),
                                        solutions)),
                        columns=['time', 'x1', 'x2'])


def normalize_training_data(dataset, ntime):
    np_dataset = dataset.to_numpy()
    norm_factor = dataset[['x1', 'x2']].max().max()

    X = np.concatenate([[np_dataset[ntime * i:ntime * (i + 1) - 1, 1:]] for i in range(np_dataset.shape[0] // ntime)],
                       axis=0)
    X /= norm_factor

    return X, norm_factor
