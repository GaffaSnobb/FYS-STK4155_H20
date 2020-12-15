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
    """
    Plot true price with predictions at different temporal locations.
    """
    dropout_rates = [0, 0.4, 0.8]
    n_dropout_rates = len(dropout_rates)
    predict_seq_len = 100
    n_predictions = 5
    prediction_shifts = [80, 50, 20, 1]
    n_prediction_shifts = len(prediction_shifts)
    scope = np.arange(predict_seq_len)

    fig, ax = plt.subplots(
        nrows = 2,
        ncols = 2,
        figsize = (9, 7)
    )

    ax = ax.ravel()
    # Limits for the ax insets.
    xlims = [[11, 21], [41, 51], [71, 81], [90, 100]]
    ylims = [[0.4, 0.6], [0.42, 0.62], [0.62, 0.82], [0.8, 1]]

    for j in range(n_prediction_shifts):
        """
        Loop over different start points of the price predictions.
        """
        axins = ax[j].inset_axes([0.04, 0.43, 0.56, 0.56])
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.set_xlim(xlims[j])
        axins.set_ylim(ylims[j])
        ax[j].indicate_inset_zoom(axins)
        ax[j].tick_params(labelsize = 15)
        # ax[j].legend()
        # ax[j].set_xlim(xlims[j])
        # ax[j].set_ylim(ylims[j])
        for i in range(n_dropout_rates):
            """
            Loop over dropout rates.
            """
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
            if i == 0:
                """
                Plot the true price only once per subplot.
                """
                ax[j].plot(
                    scope,
                    q.scaled_price[-predict_seq_len:],
                    label = "Price",
                    color = "black"
                )
                axins.plot(
                    scope,
                    q.scaled_price[-predict_seq_len:],
                    color = "black"
                )

            scaled_price = list(q.scaled_price[-predict_seq_len - n_predictions - prediction_shifts[j]:-n_predictions - prediction_shifts[j]])
            for _ in range(n_predictions):
                """
                Predict 'n_predictions' steps in the future.
                """
                sequence = to_sequences(scaled_price, predict_seq_len - 1)  # Prepare data for the network, [batch_size, sequence_length, n_features].
                prediction = q.model.predict(sequence).ravel()  # A single prediction.
                scaled_price.append(prediction)
                scaled_price.pop(0) # Remove the first data point to keep the total number of data points.

            ax[j].plot(
                scope[-n_predictions - prediction_shifts[j] - 1:-prediction_shifts[j]],
                scaled_price[-n_predictions - 1:],
                label = f"Dropout: {dropout_rates[i]}",
                linestyle = "dashed"
            )
            axins.plot(
                scope[-n_predictions - prediction_shifts[j] - 1:-prediction_shifts[j]],
                scaled_price[-n_predictions - 1:],
                linestyle = "dashed"
            )

    # Remove surplus ax labels.
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[3].set_yticklabels([])
    fig.text(s = "Days", fontsize = 15, x = 0.49, y = 0.02)
    fig.text(s = "BTC price", fontsize = 15, x = 0, y = 0.48, rotation = 90)
    fig.tight_layout(pad = 2)
    fig.savefig(fname = "../fig/price_predictions.png", dpi = 300)
    plt.show()


if __name__ == "__main__":
    plot_price_v2()
    pass