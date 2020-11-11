import numpy as np
import matplotlib.pyplot as plt
import common


def regression_relu():
    # np.random.seed(1337)
    n_data_total = 400
    x1 = np.random.uniform(0, 1, n_data_total)
    x2 = np.random.uniform(0, 1, n_data_total)
    X = np.zeros(shape=(n_data_total, 2))
    for i in range(n_data_total): X[i] = x1[i], x2[i]
    y = common.franke_function(x1, x2)

    q1 = common.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50, 25, 25),
        n_categories = 1,
        n_epochs = 300,
        batch_size = 40,
        hidden_layer_activation_function = common.relu,
        hidden_layer_activation_function_derivative = common.relu_derivative,
        output_activation_function = common.linear,
        cost_function_derivative = common.mse_derivative,
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

    q1 = common.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50, 25, 25),
        n_categories = 1,
        n_epochs = 300,
        batch_size = 40,
        hidden_layer_activation_function = common.leaky_relu,
        hidden_layer_activation_function_derivative = common.leaky_relu_derivative,
        output_activation_function = common.linear,
        cost_function_derivative = common.mse_derivative,
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

    plt.title("leaky relu")
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



def regression_vary_hidden_layer_activation_function():
    # np.random.seed(1337)
    n_data_total = 400
    x1 = np.random.uniform(0, 1, n_data_total)
    x2 = np.random.uniform(0, 1, n_data_total)
    X = np.zeros(shape=(n_data_total, 2))
    for i in range(n_data_total): X[i] = x1[i], x2[i]
    y = common.franke_function(x1, x2)

    q1 = common.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50, 25, 25),
        n_categories = 1,
        n_epochs = 300,
        batch_size = 40,
        hidden_layer_activation_function = common.sigmoid,
        hidden_layer_activation_function_derivative = common.sigmoid_derivative,
        output_activation_function = common.linear,
        cost_function_derivative = common.mse_derivative,
        verbose = True,
        debug = False,
        scaling = True)


    hidden_layer_activation_functions =\
        [common.sigmoid, common.relu, common.leaky_relu]
    hidden_layer_activation_function_derivatives =\
        [common.sigmoid_derivative, common.relu_derivative, common.leaky_relu_derivative]
    function_names = ["sigmoid", "relu", "leaky"]
    
    N = 10
    n_functions = len(hidden_layer_activation_functions)
    n_repetitions = 1   # Average to smooth the data.
    learning_rates = np.linspace(1e-3, 2e-2, N)
    mse_train = np.zeros(shape=(n_functions, N))
    mse_test = np.zeros(shape=(n_functions, N))
    r_train = np.zeros(shape=(n_functions, N))
    r_test = np.zeros(shape=(n_functions, N))

    for rep in range(n_repetitions):
        print(f"\nrepetition {rep+1} of {n_repetitions}")
        
        for i in range(n_functions):
            """
            Loop over hidden layer activation functions and their
            derivatives.
            """
            print()
            q1.hidden_layer_activation_function = \
                hidden_layer_activation_functions[i]
            q1.hidden_layer_activation_function_derivative = \
                hidden_layer_activation_function_derivatives[i]

            for j in range(N):
                print(f"{j+1} of {N}, {learning_rates[j]=}")
                q1.train_neural_network(learning_rate=learning_rates[j])
                q1.score()
                mse_train[i, j] += q1.mse_train
                mse_test[i, j] += q1.mse_test
                r_train[i, j] += q1.r_train
                r_test[i, j] += q1.r_test

    mse_train /= n_repetitions
    mse_test /= n_repetitions
    r_train /= n_repetitions
    r_test /= n_repetitions

    plt.plot(learning_rates, mse_train[0], label=f"train {function_names[0]}")
    plt.plot(learning_rates, mse_train[1], label=f"train {function_names[1]}")
    plt.plot(learning_rates, mse_train[2], label=f"train {function_names[2]}")
    # plt.plot(learning_rates, mse_test, label=f"test {hidden_layer_activation_functions[0]}")
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


if __name__ == "__main__":
    # regression_vary_hidden_layer_activation_function()
    # regression_relu()
    regression_leaky_relu()
    pass