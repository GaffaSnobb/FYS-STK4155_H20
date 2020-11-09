import numpy as np
import matplotlib.pyplot as plt
import common

def neural_network_franke():
    n_data_total = 400
    poly_degree = 3

    x1 = np.random.uniform(0, 1, n_data_total)
    x2 = np.random.uniform(0, 1, n_data_total)
    X = common.create_design_matrix_two_dependent_variables(x1,
        x2, n_data_total, poly_degree)
    y = common.franke_function(x1, x2)

    q1 = common.FFNN(
        design_matrix = X,
        true_output = y,
        hidden_layer_sizes = (50, 20, 20),
        n_categories = 1,
        n_epochs = 50,
        batch_size = 10,
        hidden_layer_activation_function = common.sigmoid,
        output_activation_function = common.linear,
        cost_function = common.cross_entropy_derivative,
        verbose = True,
        debug = False)

    N = 20
    learning_rates = np.logspace(-3, -1, N)
    # learning_rates = np.linspace(1e-5, 1, N)
    mse_train = np.zeros(N)
    mse_test = np.zeros(N)

    for i in range(N):
        q1.train_neural_network(learning_rates[i])
        mse_train[i], mse_test[i] = q1.mse

    plt.plot(learning_rates, mse_train, label="train")
    plt.plot(learning_rates, mse_test, label="test")
    plt.xlabel("learning rate")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    neural_network_franke()

    # q1 = common.FFNN(X=digits.images, y=digits.target,
    #     hidden_layer_sizes=(50, 20, 20), n_categories=10,
    #     hidden_layer_activation_function=common.sigmoid,
    #     output_activation_function=common.softmax,
    #     cost_function=common.cross_entropy_derivative,
    #     verbose=True)
    pass