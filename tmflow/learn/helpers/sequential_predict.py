from numpy import vstack
import itertools


def sequential_predict(model, X0):
    """
    Returns generator that allows making subsequent model prediction steps of the model

    :param model: pnn model
    :param X0: initial data
    :return: generator of predictions
    """
    X = X0.reshape(1, len(X0))
    while True:
        yield X
        X = model.predict(X)


def sequential_predict_batch(model, X0, size):
    """
    Returns requested number of subsequent predictions

    :param model: pnn model
    :param X0: initial data
    :param size: number of steps
    :return: vertical stack of predicted data
    """
    predict = sequential_predict(model, X0)
    res = itertools.islice(predict, size)

    return vstack(tuple(res))
