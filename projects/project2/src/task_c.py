import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import neural_network as nn
import activation_functions as af
import common


def regression_relu():
    # np.random.seed(1337)
    n_data_total = 400
    x1 = np.random.uniform(0, 1, n_data_total)
    x2 = np.random.uniform(0, 1, n_data_total)
    X = np.zeros(shape=(n_data_total, 2))
    for i in range(n_data_total): X[i] = x1[i], x2[i]
    y = common.franke_function(x1, x2)

    q1 = nn.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50, 25, 25),
        n_categories = 1,
        n_epochs = 300,
        batch_size = 40,
        hidden_layer_activation_function = af.relu,
        hidden_layer_activation_function_derivative = af.relu_derivative,
        output_activation_function = af.linear,
        cost_function_derivative = af.mse_derivative,
        verbose = True,
        debug = False,
        scaling = True)
    
    N = 10
    n_repetitions = 1   # Average to smooth the data.
    learning_rates = np.linspace(1e-3, 2e-2, N)
    mse_train = np.zeros(shape=(N))
    mse_test = np.zeros(shape=(N))
    r_train = np.zeros(shape=(N))
    r_test = np.zeros(shape=(N))

    for rep in range(n_repetitions):
        print(f"\nrepetition {rep+1} of {n_repetitions}")

        for j in range(N):
            print(f"{j+1} of {N}, {learning_rates[j]=}")
            q1.train_neural_network(learning_rate=learning_rates[j])
            q1.score()
            mse_train[j] += q1.mse_train
            mse_test[j] += q1.mse_test
            r_train[j] += q1.r_train
            r_test[j] += q1.r_test

    mse_train /= n_repetitions
    mse_test /= n_repetitions
    r_train /= n_repetitions
    r_test /= n_repetitions

    plt.title("relu")
    plt.plot(learning_rates, mse_train, label=f"train")
    plt.plot(learning_rates, mse_test, label=f"test")
    plt.xlabel("learning rates")
    plt.ylabel("mse")
    plt.legend()
    plt.show()

    # plt.plot(learning_rates, r_train, label="train")
    # plt.plot(learning_rates, r_test, label="test")
    # plt.xlabel("learning rates")
    # plt.ylabel("r_score")
    # plt.legend()
    # plt.show()


def regression_leaky_relu():
    # np.random.seed(1337)
    n_data_total = 400
    x1 = np.random.uniform(0, 1, n_data_total)
    x2 = np.random.uniform(0, 1, n_data_total)
    X = np.zeros(shape=(n_data_total, 2))
    for i in range(n_data_total): X[i] = x1[i], x2[i]
    y = common.franke_function(x1, x2)
    noise = np.random.normal(size=n_data_total)*0.1
    y += noise

    q1 = nn.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50, 25, 25),
        n_categories = 1,
        n_epochs = 20,
        batch_size = 40,
        hidden_layer_activation_function = af.leaky_relu,
        hidden_layer_activation_function_derivative = af.leaky_relu_derivative,
        output_activation_function = af.linear,
        cost_function_derivative = af.mse_derivative,
        verbose = True,
        debug = False,
        scaling = True)
    
    n_learning_rates = 6
    n_regularization_parameters = 10
    n_repetitions = 1   # Average to smooth the data.
    learning_rates = np.linspace(1e-5, 3e-4, n_learning_rates)
    regularization_parameters = np.linspace(0, 1e-3, n_regularization_parameters)
    
    mse_train = np.zeros(shape=(n_learning_rates, n_regularization_parameters))
    mse_test = np.zeros(shape=(n_learning_rates, n_regularization_parameters))
    r_train = np.zeros(shape=(n_learning_rates, n_regularization_parameters))
    r_test = np.zeros(shape=(n_learning_rates, n_regularization_parameters))

    for rep in range(n_repetitions):
        print(f"\nrepetition {rep+1} of {n_repetitions}")
        for i in range(n_learning_rates):
            for j in range(n_regularization_parameters):
                
                print(f"{j+1} of {n_regularization_parameters}, {learning_rates[i]=}")
                q1.train_neural_network(learning_rate=learning_rates[i], lambd=regularization_parameters[j])
                q1.score()
                mse_train[i, j] += q1.mse_train
                mse_test[i, j] += q1.mse_test
                r_train[i, j] += q1.r_train
                r_test[i, j] += q1.r_test

    mse_train /= n_repetitions
    mse_test /= n_repetitions
    r_train /= n_repetitions
    r_test /= n_repetitions

    ax = sns.heatmap(mse_train, linewidth=0.5, annot=True, cmap='viridis')
    ax.set_xticklabels(regularization_parameters)
    ax.set_yticklabels(learning_rates)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', rotation=0)
    ax.set_ylabel(r"$\eta$")
    ax.set_xlabel(r"$\lambda$")
    plt.show()

    # plt.title("leaky relu")
    # plt.plot(learning_rates, mse_train, label=f"train")
    # plt.plot(learning_rates, mse_test, label=f"test")
    # plt.xlabel("learning rates")
    # plt.ylabel("mse")
    # plt.legend()
    # plt.show()

    # plt.plot(learning_rates, r_train, label="train")
    # plt.plot(learning_rates, r_test, label="test")
    # plt.xlabel("learning rates")
    # plt.ylabel("r_score")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    # regression_relu()
    regression_leaky_relu()
    pass