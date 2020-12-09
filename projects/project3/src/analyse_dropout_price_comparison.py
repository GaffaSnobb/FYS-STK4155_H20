import matplotlib.pyplot as plt
import numpy as np
from lstm import CryptoPrediction


def plot_price():
    dropout_rates = [0, 0.2, 0.4, 0.6, 0.8]
    n_dropout_rates = len(dropout_rates)
    n_repetitions = 1

    Y_predicted = np.empty((59, n_dropout_rates))

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

        Y_predicted[:, i] = q.scaler.inverse_transform(q.model.predict(q.X_test)).ravel()

    y_test = q.scaler.inverse_transform(q.y_test)
    # y_train = q.scaler.inverse_transform(q.y_train)
    # y_predicted = q.scaler.inverse_transform(y_predicted)

    # # plt.plot(q.price, label="price")
    # plt.plot(y_test, label="y_test")
    # # plt.plot(y_train, label="y_train")
    # # plt.plot(np.concatenate((y_train, y_test)), label="conc")

    fig, ax = plt.subplots(figsize = (9, 7))

    ax.plot(Y_predicted[:, 0], label = f"{dropout_rates[0]=}", color = "black", linestyle = "dashed")
    ax.plot(Y_predicted[:, 2], label = f"{dropout_rates[2]=}", color = "black", linestyle = "dotted")
    ax.plot(Y_predicted[:, 4], label = f"{dropout_rates[4]=}", color = "black", linestyle = "solid")
    ax.plot(y_test, label = f"Actual", color = "grey")
    ax.set_title('BTC prediction', fontsize = 15)
    ax.set_xlabel('Time, [days]', fontsize = 15)
    ax.set_ylabel('Price, [USD]', fontsize = 15)
    ax.legend(loc='best', fontsize = 15)
    ax.grid()
    ax.tick_params(labelsize = 15)
    plt.show()


if __name__ == "__main__":
    plot_price()
    pass