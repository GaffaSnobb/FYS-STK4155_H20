import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import activation_functions as af
import neural_network as nn
import common

def classification():
    np.random.seed(1337)
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    
    q1 = nn.FFNNClassifier(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50,),
        n_categories = 10,
        n_epochs = 50,
        batch_size = 20,
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.softmax,
        cost_function_derivative = af.cross_entropy_derivative_with_softmax,
        scaling = False,
        verbose = True,
        debug = False)

    N = 10
    learning_rates = np.linspace(1e-3, 1e-2, N)
    scores = np.zeros(N)

    for i in range(N):
        q1.train_neural_network(learning_rate=learning_rates[i])
        scores[i] = q1.score(q1.X_test, q1.y_test)
        print(scores[i])

    plt.plot(learning_rates, scores)
    plt.xlabel("learning rate")
    plt.ylabel("score")
    plt.show()


def regression_vary_learning_rate():
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
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.linear,
        cost_function_derivative = af.mse_derivative,
        verbose = True,
        debug = False,
        scaling = True)

    N = 10
    n_repetitions = 1   # Average to smooth the data.
    learning_rates = np.linspace(1e-3, 2e-1, N)
    mse_train = np.zeros(N)
    mse_test = np.zeros(N)
    r_train = np.zeros(N)
    r_test = np.zeros(N)

    for rep in range(n_repetitions):
        print(f"\nrepetition {rep+1} of {n_repetitions}")
        
        for i in range(N):
            print(f"{i+1} of {N}, {learning_rates[i]=}")
            q1.train_neural_network(learning_rate=learning_rates[i])
            q1.score()
            mse_train[i] += q1.mse_train
            mse_test[i] += q1.mse_test
            r_train[i] += q1.r_train
            r_test[i] += q1.r_test

    mse_train /= n_repetitions
    mse_test /= n_repetitions
    r_train /= n_repetitions
    r_test /= n_repetitions

    plt.plot(learning_rates, mse_train, label="train")
    plt.plot(learning_rates, mse_test, label="test")
    plt.xlabel("learning rates")
    plt.ylabel("mse")
    plt.legend()
    plt.show()

    plt.plot(learning_rates, r_train, label="train")
    plt.plot(learning_rates, r_test, label="test")
    plt.xlabel("learning rates")
    plt.ylabel("r_score")
    plt.legend()
    plt.show()


def logistic():
    q1 = nn.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(),# No hidden layers for logistic!
        n_categories = 1,
        n_epochs = 300,
        batch_size = 30,
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.softmax,
        cost_function_derivative = af.mse_derivative,
        verbose = True,
        debug = False,
        scaling = False)


if __name__ == "__main__":
    classification()
    # regression_vary_learning_rate()
    pass
