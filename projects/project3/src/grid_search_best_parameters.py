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
    seq_len : int
        Sequence length for the reshaping of the data.

    dropout_rate : float
        Dropout fraction for all dropout layers.

    batch_size : int
        Number of samples per gradient update.

    epochs : int
        Number of epochs to train the model. An epoch is an
        iteration over the entire x and y data provided.

    neurons : int
        The number of neurons in each layer.

    Returns
    -------
    q.val_loss : numpy.ndarray
        Test MSE for all epochs up to the defined number of epochs.
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


def grid_search():
    """
    Perform a large grid search, varying dropout rates, sequence
    lenghts, batch_sizes, and number of neurons.
    """
    dropout_rates = [0, 0.2, 0.4, 0.6, 0.8]
    n_dropout_rates = len(dropout_rates)
    
    seq_lengths = np.arange(10, 100 + 1, 30)
    n_seq_lengths = len(seq_lengths)
    
    n_epochs = 90
    epochs = np.arange(1, n_epochs + 1, 1)
    
    batch_sizes = [2**x for x in range(2, 8 + 1, 2)]
    n_batch_sizes = len(batch_sizes)
    
    neurons = np.arange(10, 100, 20)
    n_neurons = len(neurons)

    mse = np.zeros(
        (n_dropout_rates, n_seq_lengths, n_batch_sizes, n_neurons, n_epochs),
        dtype = float)

    parallel_results = []
    for drop in range(n_dropout_rates):
        for seq in range(n_seq_lengths):
            for bat in range(n_batch_sizes):
                for neu in range(n_neurons):
                    """
                    Generate processes.  Loop over dropout rates,
                    sequence lenghts, batch_sizes, and number of
                    neurons.
                    """
                    parallel_results.append(run_network.remote(
                        seq_len = seq_lengths[seq],
                        dropout_rate = dropout_rates[drop],
                        batch_size = batch_sizes[bat],
                        epochs = n_epochs,
                        neurons = neurons[neu]
                    ))

    parallel_results = ray.get(parallel_results)
    idx = 0
    for drop in range(n_dropout_rates):
        for seq in range(n_seq_lengths):
            for bat in range(n_batch_sizes):
                for neu in range(n_neurons):
                    """
                    Extract parallel data.
                    """
                    mse[drop, seq, bat, neu] = parallel_results[idx]
                    idx += 1

    drop_min, seq_min, bat_min, neu_min, epoc_min = \
        np.unravel_index(mse.argmin(), mse.shape)
    
    print(f"min dropout: {dropout_rates[drop_min]}")
    print(f"min seq len: {seq_lengths[seq_min]}")
    print(f"min batch size: {batch_sizes[bat_min]}")
    print(f"min epoch: {np.arange(1, n_epochs+1)[epoc_min]}")
    print(f"min neuron: {neurons[neu_min]}")

    fig0, ax0 = plt.subplots(figsize = (9, 7))

    for i in range(mse.shape[0]):
        ax0.plot(epochs, mse[i, seq_min, bat_min, neu_min, :], label = f"Dropout rate: {dropout_rates[i]}")

    ax0.set_xlabel("Epochs", fontsize = 15)
    ax0.set_ylabel("MSE", fontsize = 15)
    ax0.legend(fontsize = 15)
    ax0.tick_params(labelsize = 15)
    ax0.grid()
    fig0.savefig(fname = "../fig/best_parameters_dropout_rates.png", dpi = 300)
    # plt.show()


    fig1, ax1 = plt.subplots(figsize = (9, 7))

    for i in range(mse.shape[1]):
        ax1.plot(epochs, mse[drop_min, i, bat_min, neu_min, :], label = f"Seq. length: {seq_lengths[i]}")

    ax1.set_xlabel("Epochs", fontsize = 15)
    ax1.set_ylabel("MSE", fontsize = 15)
    ax1.legend(fontsize = 15)
    ax1.tick_params(labelsize = 15)
    ax1.grid()
    fig1.savefig(fname = "../fig/best_parameters_seq_lengths.png", dpi = 300)
    # plt.show()


    fig2, ax2 = plt.subplots(figsize = (9, 7))

    for i in range(mse.shape[2]):
        ax2.plot(epochs, mse[drop_min, seq_min, i, neu_min, :], label = f"Batch size: {batch_sizes[i]}")

    ax2.set_xlabel("Epochs", fontsize = 15)
    ax2.set_ylabel("MSE", fontsize = 15)
    ax2.legend(fontsize = 15)
    ax2.tick_params(labelsize = 15)
    ax2.grid()
    fig2.savefig(fname = "../fig/best_parameters_batch_sizes.png", dpi = 300)
    # plt.show()


    fig3, ax3 = plt.subplots(figsize = (9, 7))

    for i in range(mse.shape[3]):
        ax3.plot(epochs, mse[drop_min, seq_min, bat_min, i, :], label = f"Neurons: {neurons[i]}")

    ax3.set_xlabel("Epochs", fontsize = 15)
    ax3.set_ylabel("MSE", fontsize = 15)
    ax3.legend(fontsize = 15)
    ax3.tick_params(labelsize = 15)
    ax3.grid()
    fig3.savefig(fname = "../fig/best_parameters_neurons.png", dpi = 300)
    # plt.show()


def vary_sequence_lengths():
    """
    Plot MSE as a function of sequence length.
    """
    sequence_lengths = np.arange(10, 100+1, 1)
    n_sequence_lengths = len(sequence_lengths)
    n_epochs = 90

    mse = np.empty((n_sequence_lengths, n_epochs))

    parallel_results = []
    for i in range(n_sequence_lengths):
        """
        Loop over sequence lenghts.
        """
        parallel_results.append(run_network.remote(
            seq_len = sequence_lengths[i],
            dropout_rate = 0,
            batch_size = 4,
            epochs = n_epochs,
            neurons = 70
        ))

    parallel_results = ray.get(parallel_results)
    for i in range(n_sequence_lengths):
        """
        Extract parallel work.
        """
        mse[i] = parallel_results[i]

    fig, ax = plt.subplots(figsize = (9, 7))
    ax.plot(sequence_lengths, mse[:, -1], color = "black")
    ax.grid()
    ax.tick_params(labelsize = 15)
    ax.set_ylabel("MSE", fontsize = 15)
    ax.set_xlabel("Seq. length", fontsize = 15)
    fig.savefig(fname = "../fig/vary_seq_length.png", dpi = 300)
    plt.show()


if __name__ == "__main__":
    grid_search()
    vary_sequence_lengths()
    pass
