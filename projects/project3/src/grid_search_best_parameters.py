import sys
import matplotlib.pyplot as plt
import numpy as np
import ray
from lstm import CryptoPrediction


ray.init()

@ray.remote
def run_network(seq_len, dropout_rate, batch_size, epochs, neurons):
    """
    Set up a LSTM network and train.  This function exists to be
    parallelized with ray.

    Parameters
    ----------
    dropout_rate : float
        Fraction of nodes to exclude.  Must be [0, 1].
    """
    q = CryptoPrediction(
        seq_len = seq_len,
        train_size = 0.95,
        dropout = dropout_rate,
        batch_size = batch_size,
        epochs = epochs,
        data_start = 1500,
        neurons = neurons,
        csv_path = "data/btc-usd-max.csv",
        directory = "grid_search/"
    )
    q.train_model(n_repetitions = 1)
    return q.val_loss


def stuff():
    dropout_rates = [0, 0.2, 0.4, 0.6, 0.8]
    n_dropout_rates = len(dropout_rates)
    seq_lengths = np.arange(25, 100, 15)
    n_seq_lengths = len(seq_lengths)
    n_repetitions = 1
    n_epochs = 10
    epochs = np.arange(1, n_epochs + 1, 1)
    batch_sizes = [2**x for x in range(2, 6+1)]
    n_batch_sizes = len(batch_sizes)

    mse = np.zeros((n_dropout_rates, n_seq_lengths, n_batch_sizes, n_epochs), dtype = float)

    parallel_results = []
    for drop in range(n_dropout_rates):
        for seq in range(n_seq_lengths):
            for bat in range(n_batch_sizes):
                """
                Generate processes.  Loop over dropout rates.
                """
                parallel_results.append(run_network.remote(
                    seq_len = seq_lengths[seq],
                    dropout_rate = dropout_rates[drop],
                    batch_size = batch_sizes[bat],
                    epochs = n_epochs,
                    neurons = 50
                ))

    parallel_results = ray.get(parallel_results)
    idx = 0
    for drop in range(n_dropout_rates):
        for seq in range(n_seq_lengths):
            for bat in range(n_batch_sizes):
                mse[drop, seq, bat] = parallel_results[idx]
                idx += 1

    drop_min, seq_min, bat_min, epoc_min = np.unravel_index(mse.argmin(), mse.shape)
    print(f"min dropout: {dropout_rates[drop_min]}")
    print(f"min seq len: {seq_lengths[seq_min]}")
    print(f"min batch size: {batch_sizes[bat_min]}")
    print(f"min epoch: {np.arange(1, n_epochs+1)[epoc_min]}")

    fig0, ax0 = plt.subplots(figsize = (9, 7))

    for i in range(mse.shape[0]):
        ax0.plot(epochs, mse[i, seq_min, bat_min, :], label = f"{dropout_rates[i]=}")

    ax0.set_xlabel("Epochs", fontsize = 15)
    ax0.set_ylabel("MSE", fontsize = 15)
    ax0.legend(fontsize = 15)
    ax0.tick_params(labelsize = 15)
    ax0.grid()
    fig0.savefig(fname = "../fig/best_parameters_dropout_rates.png", dpi = 300)
    # plt.show()


    fig1, ax1 = plt.subplots(figsize = (9, 7))

    for i in range(mse.shape[1]):
        ax1.plot(epochs, mse[drop_min, i, bat_min, :], label = f"{seq_lengths[i]=}")

    ax1.set_xlabel("Epochs", fontsize = 15)
    ax1.set_ylabel("MSE", fontsize = 15)
    ax1.legend(fontsize = 15)
    ax1.tick_params(labelsize = 15)
    ax1.grid()
    fig1.savefig(fname = "../fig/best_parameters_seq_lengths.png", dpi = 300)
    # plt.show()


    fig2, ax2 = plt.subplots(figsize = (9, 7))

    for i in range(mse.shape[2]):
        ax2.plot(epochs, mse[drop_min, seq_min, i, :], label = f"{batch_sizes[i]=}")

    ax2.set_xlabel("Epochs", fontsize = 15)
    ax2.set_ylabel("MSE", fontsize = 15)
    ax2.legend(fontsize = 15)
    ax2.tick_params(labelsize = 15)
    ax2.grid()
    fig2.savefig(fname = "../fig/best_parameters_batch_sizes.png", dpi = 300)
    # plt.show()



if __name__ == "__main__":
    stuff()