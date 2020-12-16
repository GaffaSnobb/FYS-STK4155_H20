import matplotlib.pyplot as plt
from lstm import CryptoPrediction
import numpy as np

activation_functions = ['tanh', 'relu', 'sigmoid']
line_style = ["dashed","dotted","solid"]
fig,ax = plt.subplots(figsize = (9, 7))


def plot(i):
    q = CryptoPrediction(
        seq_len = 100,
        train_size = 0.95,
        dropout = 0,#0.2,
        batch_size = 4,#64,
        epochs = 90,#20,
        data_start = 1500,
        neurons = 70,#50,
        hidden_activation = activation_functions[i],
        csv_path = "data/btc-usd-max.csv"
    )
    q.train_model()
    ax.plot(q.val_loss, label=activation_functions[i],color='black',linestyle=line_style[i])
    ax.tick_params(labelsize=15)
    #ax.yaxis.set_label_position('right')
    ax.ticklabel_format(useMathText=True)
    #ax.ticklabel_format(style='sci', scilimits=(-10,10))



if __name__ == "__main__":
    for i in range(len(activation_functions)):
        plot(i)
    plt.xlabel("Epochs", fontsize = 15)
    plt.ylabel("MSE", fontsize = 15)
    plt.legend(fontsize = 15)
    plt.grid()

    plt.savefig(fname = "../fig/analyse_dropout.png", dpi = 300)
    plt.show()
    """
    # ax.set_title('model loss', fontsize = 15)
    ax.set_ylabel("MSE", fontsize = 15)
    ax.set_xlabel("Epochs", fontsize = 15)
    ax.legend(fontsize = 15)
    ax.grid()
    ax.tick_params(labelsize=15)
    fig.savefig(fname = "../fig/analyse_dropout.png", dpi = 300)
    plt.show()
    """