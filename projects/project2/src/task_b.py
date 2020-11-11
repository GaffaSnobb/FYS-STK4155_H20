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
        hidden_layer_sizes=(50, 20, 20),
        n_categories = 10,
        n_epochs = 50,
        batch_size = 20,
        hidden_layer_activation_function = common.sigmoid,
        hidden_layer_activation_function_derivative = common.sigmoid_derivative,
        output_activation_function = common.softmax,
        cost_function = common.cross_entropy_derivative,
        verbose = True,
        debug = False)

    N = 10
    learning_rates = np.linspace(1e-3, 1e-2, N)
    scores = np.zeros(N)

    for i in range(N):
        q1.train_neural_network(learning_rate=learning_rates[i])
        scores[i] = q1.predict(q1.X_test)
        print(scores[i])


    plt.plot(learning_rates, scores)
    plt.xlabel("learning rate")
    plt.ylabel("score")
    plt.show()

    # q1.train_neural_network(learning_rate=1e-2)
    # q1.predict(q1.X_test)
    # print(np.argmax(q1.neuron_activation[-1], axis=1))
    # print(q1.y_test)


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
    y = x1 + x1**2

    q1 = common.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(100, 50, 25),
        n_categories = 1,
        n_epochs = 70,
        batch_size = 30,
        hidden_layer_activation_function = common.sigmoid,
        hidden_layer_activation_function_derivative = common.sigmoid_derivative,
        output_activation_function = common.linear,
        cost_function = common.mse_derivative,
        verbose = True,
        debug = False,
        scaling = True)

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
        # break


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


if __name__ == "__main__":
    # classification()
    regression()

    # for learning_rate in np.logspace(-5, 0, 8):
    #     q1.train_neural_network_single(learning_rate)
    #     score = q1.predict_single(q1.X_test)
    #     print(f"score: {score} for learning rate: {learning_rate}")
    pass