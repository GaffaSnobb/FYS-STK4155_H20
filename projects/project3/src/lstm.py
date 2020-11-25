import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation, LSTM


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
    N = len(data) - seq_len
    # d = np.zeros(shape = N, dtype = np.ndarray)

    for i in range(N):
        d.append(data[i: i + seq_len])
        # d[i] = data[i:i + seq_len]

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


def lstm():
    SEQ_LEN = 100
    TRAIN_SIZE = 0.95
    WINDOW_SIZE = SEQ_LEN - 1
    DROPOUT = 0.2
    BATCH_SIZE = 64
    EPOCHS = 50
    DATA_START = 1500   # Start slice point.
    NEURONS = 50

    state_fname = f"saved_state/{SEQ_LEN=}"
    state_fname += f"_{TRAIN_SIZE=}"
    state_fname += f"_{WINDOW_SIZE=}"
    state_fname += f"_{DROPOUT=}"
    state_fname += f"_{BATCH_SIZE=}"
    state_fname += f"_{EPOCHS=}"
    state_fname += f"_{DATA_START=}"
    state_fname += f"_{NEURONS=}.npy"

    csv_path = "data/btc-usd-max.csv"
    df = pd.read_csv(csv_path, parse_dates=['snapped_at'])
    df = df.sort_values('snapped_at')   # Sort by date.
    price = df.price.values.reshape(-1, 1)  # Reshape to fit the scaler.

    price = price[DATA_START:]

    scaler = MinMaxScaler()
    scaled_price = scaler.fit_transform(X=price)
    scaled_price = scaled_price[~np.isnan(scaled_price)].reshape(-1, 1) # Remove all NaNs.

    train_test_split_time = time.time()
    X_train, y_train, X_test, y_test =\
        train_test_split(scaled_price, SEQ_LEN, TRAIN_SIZE)
    train_test_split_time = time.time() - train_test_split_time
    print(f"{train_test_split_time = }")

    # plt.plot(np.arange(len(df.price.values)), df.price.values)
    # plt.show()

    model = Sequential()

    model.add(LSTM(
        units = NEURONS,
        return_sequences = True,
        input_shape = (WINDOW_SIZE, X_train.shape[-1]),
        activation = 'tanh'
    ))

    model.add(Dropout(rate = DROPOUT))

    model.add(LSTM(
        units = NEURONS,
        return_sequences = True,
        activation = 'tanh'
    ))

    model.add(Dropout(rate = DROPOUT))

    model.add(LSTM(
        units = NEURONS,
        return_sequences = False,
        activation = 'tanh'
    ))

    # model.add(Dropout(rate = DROPOUT))    # Dont know yet whether to use this also.

    model.add(Dense(units = 1))
    
    model.add(Activation(activation = 'linear'))

    model.compile(
        loss = 'mean_squared_error',
        optimizer = 'adam'
    )

    try:
        try:
            if sys.argv[1] and os.path.isfile(state_fname):
                choice = input("Do you really want to overwrite the state file? y/n: ")
                if choice == "y":
                    raise FileNotFoundError # Force overwrite of state.
        except IndexError: pass
        
        model.set_weights(np.load(file = state_fname, allow_pickle = True))  # Load and set weights from file.
        val_loss = np.load(file = "saved_state/val_loss.npy")
        loss = np.load(file = "saved_state/loss.npy")
    
    except FileNotFoundError:
        history = model.fit(
            x = X_train,
            y = y_train,
            epochs = EPOCHS,
            batch_size = BATCH_SIZE,
            shuffle = False,    # No shuffle for Time Series.
            validation_split = 0.1
        )
        
        val_loss = history.history['val_loss']
        loss = history.history['loss']
        np.save(file = state_fname, arr = np.array(model.get_weights()), allow_pickle = True)
        np.save(file = "saved_state/val_loss.npy", arr = val_loss)
        np.save(file = "saved_state/loss.npy", arr = loss)

    # plt.plot(loss, label="train")
    # plt.plot(val_loss, label="test")
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(loc='upper left')
    # plt.show()

    # y_predict = model.predict(X_test)
    # y_test_inverse = scaler.inverse_transform(y_test)
    # y_predict_inverse = scaler.inverse_transform(y_predict)

    # plt.plot(y_test_inverse, label="Actual Price", color='green')
    # plt.plot(y_predict_inverse, label="Predicted Price", color='red')
    # plt.title('Bitcoin price prediction')
    # plt.xlabel('Time [days]')
    # plt.ylabel('Price')
    # plt.legend(loc='best')
    # plt.show()


    # y_predict = model.predict(scaled_price)
    # y_test_inverse = scaler.inverse_transform(y_test)
    # y_predict_inverse = scaler.inverse_transform(y_predict)

    # plt.plot(y_test_inverse, label="Actual Price", color='green')
    # plt.plot(y_predict_inverse, label="Predicted Price", color='red')
    # plt.title('Bitcoin price prediction')
    # plt.xlabel('Time [days]')
    # plt.ylabel('Price')
    # plt.legend(loc='best')
    # plt.show()
    

if __name__ == "__main__":
    lstm()