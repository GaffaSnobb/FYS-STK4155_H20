import sys
import matplotlib.pyplot as plt
import numpy as np
import ray
from lstm import CryptoPrediction

ray.init()

@ray.remote
def parallel_dropout_rates(dropout_rate, n_repetitions):
    q = CryptoPrediction(
        seq_len = 100,
        train_size = 0.95,
        dropout = dropout_rate,
        batch_size = 64,
        epochs = 90,
        data_start = 1500,
        neurons = 50,
        csv_path = "data/btc-usd-max.csv",
        directory = "analyse_dropout/"
    )
    q.train_model(n_repetitions)
    return q.val_loss


def bar_label(rects, ax):
    """
    Put labels on top of each bar, indicating the height.

    From: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2g}",
                    xy = (rect.get_x() + rect.get_width() / 2, height),
                    xytext = (0, 3),  # 3 points vertical offset
                    textcoords = "offset points",
                    ha = 'center',
                    va = 'bottom')
    

def plot_dropout_rates():
    dropout_rates = [0, 0.2, 0.4, 0.6, 0.8]
    n_dropout_rates = len(dropout_rates)
    n_repetitions = 5

    mse_90 = np.zeros(n_dropout_rates)  # MSE at 50 epochs.
    mse_60 = np.zeros(n_dropout_rates)
    mse_30 = np.zeros(n_dropout_rates)

    parallel_results = []
    for dropout_rate in dropout_rates:
        """
        Generate processes.  Loop over dropout rates.
        """
        parallel_results.append(parallel_dropout_rates.remote(
            dropout_rate = dropout_rate,
            n_repetitions = n_repetitions
        ))

    fig0, ax0 = plt.subplots(figsize = (9, 7))
    for i, res in enumerate(ray.get(parallel_results)):
        """
        Execute parallel work.
        """
        if ((i%2) == 0):
            ax0.plot(res, label = f"dropout rate: {dropout_rates[i]}")
        
        mse_90[i] = res[90 - 1]
        mse_60[i] = res[60 - 1]
        mse_30[i] = res[30 - 1]

    ax0.set_ylabel("MSE", fontsize = 15)
    ax0.set_xlabel("Epochs", fontsize = 15)
    ax0.legend(fontsize = 15)
    ax0.grid()
    ax0.tick_params(labelsize = 15)
    fig0.savefig(fname = "../fig/analyse_dropout.png", dpi = 300)
    plt.show()

    fig1, ax1 = plt.subplots(figsize = (9, 7))
    x_loc = np.arange(n_dropout_rates)
    width = 0.3

    rects0 = ax1.bar(
        x = x_loc - width,
        height = mse_30,
        width = width,
        label = "30 epochs"
    )

    rects1 = ax1.bar(
        x = x_loc,
        height = mse_60,
        width = width,
        label = "60 epochs"
    )

    rects2 = ax1.bar(
        x = x_loc + width,
        height = mse_90,
        width = width,
        label = "90 epochs"
    )
        
    ax1.set_xlabel("Dropout rates", fontsize = 15)
    ax1.set_ylabel("MSE", fontsize = 15)
    ax1.tick_params(labelsize = 15)
    ax1.xaxis.set_ticks([0, *x_loc])
    ax1.xaxis.set_ticklabels([f"{x:.1f}" for x in [0, *dropout_rates]])
    ax1.legend(fontsize = 15)

    fig1.savefig(fname = "../fig/analyse_dropout_final_mse.png", dpi = 300)
    plt.show()



if __name__ == "__main__":
    plot_dropout_rates()
    pass




    # # y_predict = q.model.predict(q.scaled_price)
    # y_predict = q.model.predict(q.X_test)
    # y_test = q.scaler.inverse_transform(q.y_test)
    # y_train = q.scaler.inverse_transform(q.y_train)
    # y_predict = q.scaler.inverse_transform(y_predict)

    # # plt.plot(q.price, label="price")
    # plt.plot(y_test, label="y_test")
    # # plt.plot(y_train, label="y_train")
    # # plt.plot(np.concatenate((y_train, y_test)), label="conc")
    # plt.plot(y_predict, label="Predicted Price", color='red')
    # plt.title('Bitcoin price prediction')
    # plt.xlabel('Time [days]')
    # plt.ylabel('Price')
    # plt.legend(loc='best')
    # plt.show()