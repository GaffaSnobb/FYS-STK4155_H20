import matplotlib.pyplot as plt
import numpy as np
from lstm import CryptoPrediction, to_sequences


def plot_price_predictions(debug = True):
    """
    Plot true price and predictions at different temporal locations.
    """
    dropout_rates = [0, 0.4, 0.8]
    n_dropout_rates = len(dropout_rates)
    predict_seq_len = 100
    n_predictions = 5   # Amount of days to predict.
    prediction_shifts = [80, 50, 20, 1] # Shift value of where to start prediction.
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

    predict_mse = np.empty(((n_dropout_rates + 1)*n_prediction_shifts, 3))
    predict_mse_idx = 0
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

                if debug:
                    """
                    Plot red dots for the places where the price
                    comparison takes place.
                    """
                    scope_tmp = scope[-n_predictions - prediction_shifts[j] - 1:-prediction_shifts[j]]
                    axins.plot(
                        scope_tmp[1],
                        q.scaled_price[-n_predictions - prediction_shifts[j]],
                        "r."
                    )
                    axins.plot(
                        scope_tmp[3],
                        q.scaled_price[-n_predictions - prediction_shifts[j] + 2],
                        "r."
                    )
                    axins.plot(
                        scope_tmp[5],
                        q.scaled_price[-n_predictions - prediction_shifts[j] + 4],
                        "r."
                    )

                axins.plot(
                    scope,
                    q.scaled_price[-predict_seq_len:],
                    color = "black"
                )
                
                predict_mse[predict_mse_idx] = \
                    (q.scaled_price[-n_predictions - prediction_shifts[j]],
                    q.scaled_price[-n_predictions - prediction_shifts[j] + 2],
                    q.scaled_price[-n_predictions - prediction_shifts[j] + 4])
                predict_mse_idx += 1

            scaled_price = list(q.scaled_price[-predict_seq_len - n_predictions - prediction_shifts[j]:-n_predictions - prediction_shifts[j]])
            for _ in range(n_predictions):
                """
                Predict 'n_predictions' steps in the future.
                """
                sequence = to_sequences(scaled_price, predict_seq_len - 1)  # Prepare data for the network, [batch_size, sequence_length, n_features].
                prediction = q.model.predict(sequence).ravel()  # A single prediction.
                scaled_price.append(prediction)
                scaled_price.pop(0) # Remove the first data point to retain the total number of data points.

            predict_mse[predict_mse_idx] = scaled_price[-5], scaled_price[-3], scaled_price[-1]
            predict_mse_idx += 1

            ax[j].plot(
                scope[-n_predictions - prediction_shifts[j] - 1:-prediction_shifts[j]],
                scaled_price[-n_predictions - 1:],
                label = f"Dropout: {dropout_rates[i]}",
                linestyle = "dashed"
            )
            if debug:
                """
                Plot red dots for the places where the price
                comparison takes place.
                """
                scope_tmp = scope[-n_predictions - prediction_shifts[j] - 1:-prediction_shifts[j]]
                axins.plot(
                    scope_tmp[1],
                    scaled_price[-5],
                    "r."
                )
                axins.plot(
                    scope_tmp[3],
                    scaled_price[-3],
                    "r."
                )
                axins.plot(
                    scope_tmp[5],
                    scaled_price[-1],
                    "r."
                )
            
            axins.plot(
                scope[-n_predictions - prediction_shifts[j] - 1:-prediction_shifts[j]],
                scaled_price[-n_predictions - 1:],
                linestyle = "dashed"
            )

        # ax[j].legend()
    
    # Remove surplus ax labels.
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[3].set_yticklabels([])
    fig.text(s = "Days", fontsize = 15, x = 0.49, y = 0.02)
    fig.text(s = "BTC price [scaled]", fontsize = 15, x = 0, y = 0.42, rotation = 90)
    fig.tight_layout(pad = 2)
    fig.savefig(fname = "../fig/price_predictions.png", dpi = 300)
    # plt.show()

    # Print table of data.
    idx = 0
    for _ in range(n_prediction_shifts):
        print(f"\nTrue: {predict_mse[idx]}")
        idx += 1

        for i in range(n_dropout_rates):
            print(f"drop={dropout_rates[i]}: {predict_mse[idx]}")
            idx += 1


if __name__ == "__main__":
    plot_price_predictions()
    pass