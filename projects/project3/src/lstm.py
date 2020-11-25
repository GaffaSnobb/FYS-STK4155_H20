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
        # d[i] = data[i:i + seq_len]
        d.append(data[i: i + seq_len])

    return np.array(d)
    # return d


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

    X_test = data[num_train:, :-1, :]
    X_train = data[:num_train, :-1, :]
    
    y_test = data[num_train:, -1, :]
    y_train = data[:num_train, -1, :]

    return X_train, y_train, X_test, y_test


class CryptoPrediction:
    def __init__(self,
            seq_len = 100,
            train_size = 0.95,
            dropout = 0.2,
            batch_size = 64,
            epochs = 50,
            data_start = 0,
            neurons = 50
        ):
        """
        Parameters
        ----------
        seq_len : int
            Sequence length for the reshaping of the data.

        train_size : float
            Should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train split.

        dropout : float
            Dropout fraction for all dropout layers.

        batch_size : int
            Number of samples per gradient update.

        epochs : int
            Number of epochs to train the model. An epoch is an
            iteration over the entire x and y data provided.

        data_start : int
            Start index of data slice.

        neurons : int
            The number of neurons in each layer.
        """
        self.seq_len = seq_len
        self.train_size = train_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_start = data_start
        self.neurons = neurons
        window_size = self.seq_len - 1
        self.window_size = window_size

        self.state_fname = f"saved_state/{seq_len=}"
        self.state_fname += f"_{train_size=}"
        self.state_fname += f"_{window_size=}"
        self.state_fname += f"_{dropout=}"
        self.state_fname += f"_{batch_size=}"
        self.state_fname += f"_{epochs=}"
        self.state_fname += f"_{data_start=}"
        self.state_fname += f"_{neurons=}.npy"


    def _initial_state(self, csv_path = "data/btc-usd-max.csv"):
        """
        Set up the initial state of the data.  Read data from csv.  Sort
        dataframe and reshape price array to column vector.  Scale the
        data.  Split into train and test sets.

        Parameters
        ----------
        csv_path : str
            Path to csv file.
        """
        df = pd.read_csv(csv_path, parse_dates=['snapped_at'])
        df = df.sort_values('snapped_at')   # Sort by date.
        self.price = df.price.values.reshape(-1, 1)  # Reshape to fit the scaler.
        self.price = self.price[self.data_start:] # Slice dataset.

        self.scaler = MinMaxScaler()
        self.scaled_price = self.scaler.fit_transform(X = self.price)
        self.scaled_price = self.scaled_price[~np.isnan(self.scaled_price)] # Remove all NaNs.
        self.scaled_price = self.scaled_price.reshape(-1, 1)    # Reshape to column.

        train_test_split_time = time.time()
        self.X_train, self.y_train, self.X_test, self.y_test =\
            train_test_split(self.scaled_price, self.seq_len, self.train_size)
        train_test_split_time = time.time() - train_test_split_time
        print(f"{train_test_split_time = }")

        # plt.plot(np.arange(len(df.price.values)), df.price.values)
        # plt.show()


    def create_model(self,
            hidden_activation = "tanh",
            loss_function = "mean_squared_error"
        ):
        """
        Set up the model.  Create layers.
        """
        self._initial_state()
        self.model = Sequential()

        self.model.add(LSTM(
            units = self.neurons,
            return_sequences = True,
            input_shape = (self.window_size, self.X_train.shape[-1]),
            activation = hidden_activation
        ))

        self.model.add(Dropout(rate = self.dropout))

        self.model.add(LSTM(
            units = self.neurons,
            return_sequences = True,
            activation = hidden_activation
        ))

        self.model.add(Dropout(rate = self.dropout))

        self.model.add(LSTM(
            units = self.neurons,
            return_sequences = False,
            activation = hidden_activation
        ))

        # self.model.add(Dropout(rate = self.dropout))    # Dont know yet whether to use this also.

        self.model.add(Dense(units = 1))
        
        self.model.add(Activation(activation = 'linear'))

        self.model.compile(
            loss = loss_function,
            optimizer = 'adam'
        )

    
    def train_model(self):
        """
        Train the model.  If state data is already saved, no training
        will be performed.
        """
        self.create_model()
        try:
            """
            Try to load the data from file.
            """
            try:
                """
                Input handling for cml argument. Try to index
                sys.argv[1]. 
                """
                if sys.argv[1] and os.path.isfile(self.state_fname):
                    """
                    If sys.arv[1] is true, and file exists, prompt to
                    overwrite saved state.
                    """
                    choice = input("Do you really want to overwrite the state file? y/n: ")
                    if choice == "y":
                        """
                        If file overwrite is allowed by the user, raise
                        a FileNotFoundError to force the except to kick
                        in.
                        """
                        raise FileNotFoundError
            
            except IndexError:
                """
                If sys.argv[1] cannot be indexed, then no cml argument
                is given, and the program will try to load the state
                from file.
                """
                pass
            
            # Load and set weights from file.
            self.model.set_weights(np.load(
                file = self.state_fname,
                allow_pickle = True
                ))
            val_loss = np.load(file = "saved_state/val_loss.npy")
            loss = np.load(file = "saved_state/loss.npy")
            print(f"State loaded from file '{self.state_fname}''.")
        
        except FileNotFoundError:
            """
            If the trained state cannot be loaded from file, generate
            it.
            """
            history = self.model.fit(
                x = self.X_train,
                y = self.y_train,
                epochs = self.epochs,
                batch_size = self.batch_size,
                shuffle = False,    # No shuffle for Time Series.
                validation_split = 0.1
            )
            
            val_loss = history.history['val_loss']
            loss = history.history['loss']
            np.save(
                file = self.state_fname,
                arr = np.array(self.model.get_weights()), allow_pickle = True
            )
            np.save(file = "saved_state/val_loss.npy", arr = val_loss)
            np.save(file = "saved_state/loss.npy", arr = loss)
            print(f"State saved to file '{self.state_fname}''.")


    def plot(self):
        # plt.plot(loss, label="train")
        # plt.plot(val_loss, label="test")
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(loc='upper left')
        # plt.show()

        y_predict = self.model.predict(self.X_test)
        y_test_inverse = self.scaler.inverse_transform(self.y_test)
        y_predict_inverse = self.scaler.inverse_transform(y_predict)

        plt.plot(y_test_inverse, label="Actual Price", color='green')
        plt.plot(y_predict_inverse, label="Predicted Price", color='red')
        plt.title('Bitcoin price prediction')
        plt.xlabel('Time [days]')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.show()


        # # y_predict = self.model.predict(self.scaled_price)
        # y_test = self.scaler.inverse_transform(self.y_test)
        # y_train = self.scaler.inverse_transform(self.y_train)
        # # y_predict = self.scaler.inverse_transform(y_predict)

        # plt.plot(self.price, label="price")
        # plt.plot(y_test, label="y_test")
        # plt.plot(y_train, label="y_train")
        # plt.plot(np.concatenate((y_train, y_test)), label="conc")
        # # plt.plot(y_predict, label="Predicted Price", color='red')
        # plt.title('Bitcoin price prediction')
        # plt.xlabel('Time [days]')
        # plt.ylabel('Price')
        # plt.legend(loc='best')
        # plt.show()
    

if __name__ == "__main__":
    q = CryptoPrediction()
    q.train_model()
    q.plot()