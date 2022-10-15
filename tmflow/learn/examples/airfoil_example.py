import matplotlib.pyplot as plt
import time as tt
from keras.models import Sequential
from keras import optimizers
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from tmflow.learn.kronecker_polynomial_layer import KroneckerPolynomialLayer
from airfoil_data_preparation import prepare_airfoil_data


def create_polynomial_model(order, output_dim):
    model = Sequential()
    model.add(KroneckerPolynomialLayer(output_dim=output_dim, order=order, initial_weights=None))
    optimizer = optimizers.Adam(learning_rate=0.05)

    model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer, metrics=['mae', 'mape'])

    return model


def fit_polynomial_model(model, X_train, Y_train, num_epoch, batch_size):
    start = tt.time()
    model.fit(X_train, Y_train, epochs=num_epoch, batch_size=batch_size, shuffle=False, verbose=1)
    end = tt.time()
    print(poly_net.get_weights())
    print(f'PNN is built in {end-start} seconds')
    return model


def validate_results(model, X_data, Y_data):
    samples_size = len(X_data)
    Y_model = model.predict(X_data)

    print('MSE  = ', mean_squared_error(Y_data, Y_model))
    print('MAE  = ', mean_absolute_error(Y_data, Y_model))
    print('MAPE = ', mean_absolute_percentage_error(Y_data, Y_model))

    t = range(samples_size)
    plt.scatter(t, Y_data.reshape(1, -1))
    plt.scatter(t, Y_model.reshape(1, -1))
    plt.show()


if __name__ == "__main__":
    poly_net = create_polynomial_model(output_dim=1, order=3)
    X_train, X_test, Y_train, Y_test = prepare_airfoil_data()
    poly_net = fit_polynomial_model(poly_net, X_train, Y_train, num_epoch=12000, batch_size=500)
    validate_results(poly_net, X_test, Y_test)
