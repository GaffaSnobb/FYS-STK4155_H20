import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import common

def classification():
    np.random.seed(1337)
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    
    q1 = common.FFNNClassifier(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50,),
        n_categories = 10,
        n_epochs = 50,
        batch_size = 20,
        hidden_layer_activation_function = common.sigmoid,
        hidden_layer_activation_function_derivative = common.sigmoid_derivative,
        output_activation_function = common.softmax,
        cost_function_derivative = common.cross_entropy_derivative_with_softmax,
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


def regression():
    np.random.seed(1337)
    n_data_total = 400
    # x1 = np.random.uniform(0, 1, n_data_total)
    x1 = np.linspace(0, 1, n_data_total)
    # x2 = np.random.uniform(0, 1, n_data_total)
    # X = np.zeros(shape=(n_data_total, 2))
    # for i in range(n_data_total): X[i] = x1[i], x2[i]
    # y = common.franke_function(x1, x2)

    X = x1
    y = x1 + x1**2 + np.random.normal(size=n_data_total)*0.1

    q1 = common.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50, 50),
        n_categories = 1,
        n_epochs = 300,
        batch_size = 30,
        hidden_layer_activation_function = common.sigmoid,
        hidden_layer_activation_function_derivative = common.sigmoid_derivative,
        output_activation_function = common.linear,
        cost_function_derivative = common.mse_derivative,
        verbose = True,
        debug = False,
        scaling = False)


    # q1.train_neural_network(learning_rate=0.06)
    # q1.X_selection = q1.X_test
    # q1.feedforward()
    # r_test = 1 - np.sum((q1.y_test - q1.neuron_input[-1])**2)/np.sum((q1.y_test - np.mean(q1.y_test))**2)
    # print(f"{r_test=}")
    # print("numerator test: ", np.sum((q1.y_test - q1.neuron_input[-1])**2))
    # print("denominator test ", np.sum((q1.y_test - np.mean(q1.y_test))**2))
    # print(f"{q1.y_test.shape=}")
    # print(f"{q1.neuron_input[-1].shape=}")
    # plt.plot(q1.X_test, q1.neuron_input[-1], "r.",label="test")
    
    # q1.X_selection = q1.X_train
    # q1.feedforward()
    # r_train = 1 - np.sum((q1.y_train - q1.neuron_input[-1])**2)/np.sum((q1.y_train - np.mean(q1.y_train))**2)
    # print(f"{r_train=}")


    # plt.plot(q1.X_train, q1.neuron_input[-1], "b.",label="train")
    # plt.plot(X, y, label="actual")
    # plt.legend()
    # plt.show()


    N = 10
    learning_rates = np.logspace(-3, -1, N)
    mse_train = np.zeros(N)
    mse_test = np.zeros(N)
    r_train = np.zeros(N)
    r_test = np.zeros(N)

    for i in range(N):
        q1.train_neural_network(learning_rate=learning_rates[i])
        q1.score()
        mse_train[i] = q1.mse_train
        mse_test[i] = q1.mse_test
        r_train[i] = q1.r_train
        r_test[i] = q1.r_test


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
    q1 = common.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(),# No hidden layers for logistic!
        n_categories = 1,
        n_epochs = 300,
        batch_size = 30,
        hidden_layer_activation_function = common.sigmoid,
        hidden_layer_activation_function_derivative = common.sigmoid_derivative,
        output_activation_function = common.softmax,
        cost_function_derivative = common.mse_derivative,
        verbose = True,
        debug = False,
        scaling = False)

if __name__ == "__main__":
    classification()
    # regression()
    pass