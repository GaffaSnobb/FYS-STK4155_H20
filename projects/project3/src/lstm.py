import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

def to_sequences(data, seq_len):
    """
    From https://towardsdatascience.com/cryptocurrency-price-prediction-using-lstms-tensorflow-for-hackers-part-iii-264fcdbccd3f.

    Shape the data into LSTM friendly shape.  Desired shape is
    [batch_size, sequence_length, n_features].

    Parameters
    ----------
    data : numpy.ndarray
        Data to be split into sequences.

    seq_len : int
        The length of each sequence.
    """
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)


def train_test_split(data_raw, seq_len, train_size):
    """
    From https://towardsdatascience.com/cryptocurrency-price-prediction-using-lstms-tensorflow-for-hackers-part-iii-264fcdbccd3f.

    Parameters
    ----------
    data_raw : numpy.ndarray
        Scaled price data as column vector.

    seq_len : int
        Desired length of each sequence.

    train_size : float
        Should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the train split.
    """
    data = to_sequences(data_raw, seq_len)

    num_train = int(train_size*data.shape[0])  # Number of training data.

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


def lol():
    csv_path = "data/btc-usd-max.csv"
    df = pd.read_csv(csv_path, parse_dates=['snapped_at'])
    df = df.sort_values('snapped_at')   # Sort by date.
    price = df.price.values.reshape(-1, 1)  # Reshape to fit the scaler.

    scaler = MinMaxScaler()
    scaled_price = scaler.fit_transform(X=price)
    scaled_price = scaled_price[~np.isnan(scaled_price)].reshape(-1, 1) # Remove all NaNs.

    X_train, y_train, X_test, y_test =\
        train_test_split(scaled_price, seq_len = 100, train_size = 0.95)

    

if __name__ == "__main__":
    lol()