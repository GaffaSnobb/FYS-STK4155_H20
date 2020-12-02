import matplotlib.pyplot as plt
from lstm import CryptoPrediction

def plot():
    q = CryptoPrediction(
        seq_len = 100,
        train_size = 0.95,
        dropout = 0.2,
        batch_size = 64,
        epochs = 20,
        data_start = 1500,
        neurons = 50,
        csv_path = "data/btc-usd-max.csv"
    )
    q.train_model()

    fig, ax = plt.subplots(figsize = (9, 7))
    ax.plot(q.val_loss, label="test")
    # ax.set_title('model loss', fontsize = 15)
    ax.set_ylabel("MSE", fontsize = 15)
    ax.set_xlabel("Epochs", fontsize = 15)
    ax.legend(fontsize = 15)
    ax.grid()
    ax.tick_params(labelsize=15)
    fig.savefig(fname = "../fig/analyse_dropout.png", dpi = 300)
    plt.show()


if __name__ == "__main__":
    plot()