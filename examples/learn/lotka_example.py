import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras import optimizers
import os

from tmflow.learn import KroneckerPolynomialLayer
from tmflow.utils.weights import convert_initial_weights
from tmflow.learn.helpers import sequential_predict_batch

from lotka_data_preparation import init_lotka_equation, generate_training_data, normalize_training_data


def plot_results(time_points, reference_values, predicted_values, title):
    assert (len(reference_values) == len(predicted_values))

    line_count = len(reference_values)
    N = len(time_points)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        for j in range(line_count):
            ax.plot(time_points, reference_values[j][:N, i], '--', alpha=.6)
        ax.set_prop_cycle(None)

    # plotting a graph with value from pnn model
    for i, ax in enumerate(axes.flat):
        for j in range(line_count):
            ax.plot(time_points, predicted_values[j][:N, i])
        ax.set_prop_cycle(None)

    fig.suptitle(title, fontsize='large')
    plt.show()


def get_trained_pnn(use_initial_weights: bool, batch_size=1, epochs_num=2):
    if use_initial_weights:
        filename = os.path.dirname(__file__) + '/train_data/initial_weights_ord4_h0.01.pkl'
        init_weights = convert_initial_weights(dim, order, filename, norm_factor)
    else:
        init_weights = None

    # create model
    pnn = Sequential()
    pnn.add(KroneckerPolynomialLayer(dim, order, init_weights))

    opt = keras.optimizers.Adamax(learning_rate=0.02, beta_1=0.99,
                                  beta_2=0.99999, epsilon=1e-1, decay=0.0)
    pnn.compile(loss='mean_squared_error', optimizer=opt)

    # run training
    pnn.fit(input_data, output_data, epochs=epochs_num, batch_size=batch_size, verbose=1)

    return pnn


if __name__ == '__main__':
    # set parameters
    dim = 2
    order = 4
    dt = 0.01
    trajectory_len = 101

    # init equation
    equation = init_lotka_equation(a=0.5, b=0.9, c=1.0)

    # generate data
    training_data = generate_training_data(equation, trajectory_len, dt)

    # normalize data
    X, norm_factor = normalize_training_data(training_data, trajectory_len)

    # composing the dataset
    input_data = np.concatenate([X[i, :-1, :] for i in range(X.shape[0])], axis=0)
    output_data = np.concatenate([X[i, 1:, :] for i in range(X.shape[0])], axis=0)

    # create models
    pnn_with_initial_weight = get_trained_pnn(use_initial_weights=True, batch_size=trajectory_len)
    pnn_zero_initialized = get_trained_pnn(use_initial_weights=False, batch_size=trajectory_len)

    # print result
    nums = [0, 23, 54, 151, 332]
    reference_values_for_plot = [X[num] for num in nums]

    N = trajectory_len - 1
    time1 = np.arange(N) * dt

    X_predicted_with_initial_weight = [sequential_predict_batch(pnn_with_initial_weight, value[0], N)
                                       for value in reference_values_for_plot]

    X_predicted_zero_initialized = [sequential_predict_batch(pnn_zero_initialized, value[0], N)
                                    for value in reference_values_for_plot]

    plot_results(time1, reference_values_for_plot, X_predicted_with_initial_weight, 'With precomputed initial weights')
    plot_results(time1, reference_values_for_plot, X_predicted_zero_initialized, 'Zero initialized')
