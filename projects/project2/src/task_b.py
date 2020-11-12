import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from sklearn import datasets
import activation_functions as af
import neural_network as nn
import common

a_good_learning_rate = 0.09316326530612246


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
    noise = np.random.normal(size=n_data_total)*0.1
    y += noise

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

    N = 50
    n_repetitions = 5   # Average to smooth the data.
    learning_rates = np.linspace(0.005, 0.125, N)
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

    min_train_idx = np.argmin(mse_train)
    min_test_idx = np.argmin(mse_test)

    print(f"min. mse for train at learning_rate[{min_train_idx}]={learning_rates[min_train_idx]}")
    print(f"min. mse for test at learning_rate[{min_test_idx}]={learning_rates[min_test_idx]}")

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


def regression_vary_regularization_parameters():
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
        n_epochs = 300,
        batch_size = 40,
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.linear,
        cost_function_derivative = af.mse_derivative,
        verbose = True,
        debug = False,
        scaling = True)

    n_regularization_parameters = 100
    n_repetitions = 20   # Average to smooth the data.
    regularization_parameters = np.linspace(0, 1e-3, n_regularization_parameters)
    mse_train = np.zeros(n_regularization_parameters)
    mse_test = np.zeros(n_regularization_parameters)
    r_train = np.zeros(n_regularization_parameters)
    r_test = np.zeros(n_regularization_parameters)

    for rep in range(n_repetitions):
        print(f"\nrepetition {rep+1} of {n_repetitions}")
        
        for i in range(n_regularization_parameters):
            print(f"{i+1} of {n_regularization_parameters}, {regularization_parameters[i]=}")
            q1.train_neural_network(learning_rate=a_good_learning_rate, lambd=regularization_parameters[i])
            q1.score()
            mse_train[i] += q1.mse_train
            mse_test[i] += q1.mse_test
            r_train[i] += q1.r_train
            r_test[i] += q1.r_test

    mse_train /= n_repetitions
    mse_test /= n_repetitions
    r_train /= n_repetitions
    r_test /= n_repetitions

    min_train_idx = np.argmin(mse_train)
    min_test_idx = np.argmin(mse_test)

    print(f"min. mse for train at regularization[{min_train_idx}]={regularization_parameters[min_train_idx]}")
    print(f"min. mse for test at regularization[{min_test_idx}]={regularization_parameters[min_test_idx]}")

    plt.plot(regularization_parameters, mse_train, label="train")
    plt.plot(regularization_parameters, mse_test, label="test")
    plt.xlabel("regularization parameters")
    plt.ylabel("mse")
    plt.legend()
    plt.show()

    plt.plot(regularization_parameters, r_train, label="train")
    plt.plot(regularization_parameters, r_test, label="test")
    plt.xlabel("regularization parameters")
    plt.ylabel("r_score")
    plt.legend()
    plt.show()


def regression_vary_learning_rate_and_regularization_parameter():
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
        n_epochs = 50,
        batch_size = 40,
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.linear,
        cost_function_derivative = af.mse_derivative,
        verbose = True,
        debug = False,
        scaling = True)

    n_learning_rates = 2
    n_regularization_parameters = 2
    n_repetitions = 1   # Average to smooth the data.
    learning_rates = np.linspace(0.005, 0.125, n_learning_rates)
    regularization_parameters = np.linspace(0, 1e-3, n_regularization_parameters)
    
    mse_train = np.zeros(shape=(n_learning_rates, n_regularization_parameters))
    mse_test = np.zeros(shape=(n_learning_rates, n_regularization_parameters))
    r_train = np.zeros(shape=(n_learning_rates, n_regularization_parameters))
    r_test = np.zeros(shape=(n_learning_rates, n_regularization_parameters))

    for rep in range(n_repetitions):
        print(f"\nrepetition {rep+1} of {n_repetitions}")
        
        for i in range(n_learning_rates):
            for j in range(n_regularization_parameters):
                print(f"{i+1} of {n_learning_rates}, {learning_rates[i]=}")
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

    idx = np.unravel_index(np.argmin(mse_train), mse_train.shape)
    ax = sns.heatmap(mse_train, linewidth=0.5, annot=True, cmap='viridis')
    ax.set_xticklabels(regularization_parameters)
    ax.set_yticklabels(learning_rates)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', rotation=0)
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


def regression_compare_neural_network_and_ols_ridge():
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


if __name__ == "__main__":
    # classification()
    # regression_vary_learning_rate()
    # regression_vary_regularization_parameters()
    regression_vary_learning_rate_and_regularization_parameter()
    # regression_compare_neural_network_and_ols_ridge()
    pass
