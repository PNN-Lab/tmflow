import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def prepare_airfoil_data():
    filepath = Path(__file__).parent / "train_data/AirfoilSelfNoise.csv"
    df = pd.read_csv(filepath)
    return _data_train_test_split(df)


def _data_train_test_split(data_frame):
    X = np.array(data_frame.loc[:, ["f", "alpha", "c", "U_infinity", "delta"]])
    Y = np.array(data_frame.loc[:, ["SSPL"]])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_test, Y_train, Y_test
