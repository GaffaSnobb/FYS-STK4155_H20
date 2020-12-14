import sys
import matplotlib.pyplot as plt
import numpy as np
from lstm import CryptoPrediction, to_sequences


def plot_price():
    dropout_rates = [0, 0.2, 0.4, 0.6, 0.8]
    n_dropout_rates = len(dropout_rates)
    n_repetitions = 1

    days = 59
    end_day = days - 30
    Y_predicted = np.empty((days, n_dropout_rates))

    for i in range(n_dropout_rates):

        q = CryptoPrediction(
            seq_len = 100,
            train_size = 0.95,
            dropout = dropout_rates[i],
            batch_size = 64,
            epochs = 90,
            data_start = 1500,
            neurons = 50,
            csv_path = "data/btc-usd-max.csv",
            directory = "analyse_dropout/"
        )
        q.train_model(n_repetitions)

        Y_predicted[:end_day, i] = q.scaler.inverse_transform(q.model.predict(q.X_test[:end_day])).ravel()
        print(f"{q.X_test[:end_day].shape=}")
        print(f"{Y_predicted[:end_day, i].shape=}")
        print(f"{to_sequences(data = Y_predicted[:end_day, i].reshape(-1, 1), seq_len = 1).shape=}")
        

        sys.exit()
        for j in range(days - end_day):
            next_data = to_sequences(data = Y_predicted[:end_day + j, i].reshape(-1, 1), seq_len = 1)
            Y_predicted[:end_day + 1 + j, i] = q.model.predict(next_data).ravel()

    y_test = q.scaler.inverse_transform(q.y_test)


    fig, ax = plt.subplots(figsize = (9, 7))
    # ax.plot(Y_predicted[:, 0], label = f"Dropout rate: {dropout_rates[0]}", color = "black", linestyle = "dashed")
    # ax.plot(Y_predicted[:, 2], label = f"Dropout rate: {dropout_rates[2]}", color = "black", linestyle = "dotted")
    ax.plot(Y_predicted[:, 4], label = f"Dropout rate: {dropout_rates[4]}", color = "black", linestyle = "solid")
    ax.plot(y_test, label = f"Actual", color = "grey")
    ax.set_title('BTC prediction', fontsize = 15)
    ax.set_xlabel('Time, [days]', fontsize = 15)
    ax.set_ylabel('Price, [USD]', fontsize = 15)
    ax.legend(loc='best', fontsize = 15)
    ax.grid()
    ax.tick_params(labelsize = 15)
    fig.savefig(fname = "../fig/analyse_dropout_price_comparison.png", dpi = 300)
    plt.show()


def plot_price_v2():

    dropout_rates = [0, 0.2, 0.4, 0.6, 0.8]
    n_dropout_rates = len(dropout_rates)
    how_many = 100
    n_predictions = 10
    scope = np.arange(how_many)

    for i in range(n_dropout_rates):

        q = CryptoPrediction(
            seq_len = 100,
            train_size = 0.95,
            dropout = dropout_rates[i],
            batch_size = 64,
            epochs = 90,
            data_start = 1500,
            neurons = 50,
            csv_path = "data/btc-usd-max.csv",
            directory = "analyse_dropout/"
        )
        q.train_model()

        
        scaled_price = list(q.scaled_price[-how_many - n_predictions:-n_predictions])
        for _ in range(n_predictions):
            sequence = to_sequences(scaled_price, how_many - 1)
            prediction = q.model.predict(sequence).ravel()
            scaled_price.append(prediction)
            scaled_price.pop(0)

        plt.plot(scope[-n_predictions - 1:], scaled_price[-n_predictions - 1:], label = f"{dropout_rates[i]=}", linestyle = "dashed")
    
    plt.plot(scope, q.scaled_price[-how_many:], label = "true")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_price_v2()
    pass