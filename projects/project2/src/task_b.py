import time
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
    """
    Only used for testing the neural network, not directly related to
    task_b.
    """
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
    """
    Results from this function are not directly used in task b.
    """
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
    """
    Results from this function are not directly used in task b.
    """
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
    """
    This is handled by task c.
    """
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


def regression_compare_neural_network_and_ols_ridge():
    """
    Actual results for task b.  Generate a Franke data set and solve it
    with the linear regression code from project 1.  Also solve it with
    the neural network.  Print detailed comparison data.

    (Disregard the cross validation data.  Something is off with the
    r score, so we'll just use bootstrapping instead.)
    """
    # np.random.seed(1337)
    n_data_total = 400
    x1 = np.random.uniform(0, 1, n_data_total)
    x2 = np.random.uniform(0, 1, n_data_total)
    noise = np.random.normal(size=n_data_total)*0.1
    y = common.franke_function(x1, x2) + noise

    # Data for linear regression:
    n_repetitions = 50  # Repeat and average the linear regression calculation.
    polynomial_degree = 5
    ridge_parameter = 1e-3
    design_matrix = common.create_design_matrix_two_dependent_variables(
        x1, x2, n_data_total, polynomial_degree)

    linear_regression = common.Regression(
        design_matrix = design_matrix,
        true_output = y,
        polynomial_degree = polynomial_degree,
        scale = True)

    mse_train_cv = 0
    mse_test_cv = 0
    r_score_train_cv = 0
    r_score_test_cv = 0

    mse_train_boot = 0
    mse_test_boot = 0
    r_score_train_boot = 0
    r_score_test_boot = 0

    mse_train_boot_ridge = 0
    mse_test_boot_ridge = 0
    r_score_train_boot_ridge = 0
    r_score_test_boot_ridge = 0
    
    for _ in range(n_repetitions):
        """
        Repeat and average the data.  Cross validation ols.
        """
        linear_regression.cross_validation(
            degree = polynomial_degree,
            folds = 5,
            lambd = 0,  # Ridge.
            alpha = 0,  # Lasso.
            shuffle = False)

        mse_train_cv += linear_regression.mse_train_cv
        mse_test_cv += linear_regression.mse_test_cv
        r_score_train_cv += linear_regression.r_score_train_cv
        r_score_test_cv += linear_regression.r_score_test_cv

    for _ in range(n_repetitions):
        """
        Repeat and average the data.  Bootstrapping ols.
        """
        linear_regression.bootstrap(
            degree = polynomial_degree,
            n_bootstraps = 50,
            lambd = 0,  # Ridge.
            alpha = 0)  # Lasso.
        
        mse_train_boot += linear_regression.mse_train_boot
        mse_test_boot += linear_regression.mse_test_boot
        r_score_train_boot += linear_regression.r_score_train_boot
        r_score_test_boot += linear_regression.r_score_test_boot

    linear_regression_time = time.time()
    for rep in range(n_repetitions):
        """
        Repeat and average the data.  Bootstrapping ridge.
        """
        linear_regression.bootstrap(
            degree = polynomial_degree,
            n_bootstraps = 50,
            lambd = ridge_parameter,  # Ridge.
            alpha = 0)  # Lasso.

        mse_train_boot_ridge += linear_regression.mse_train_boot
        mse_test_boot_ridge += linear_regression.mse_test_boot
        r_score_train_boot_ridge += linear_regression.r_score_train_boot
        r_score_test_boot_ridge += linear_regression.r_score_test_boot
        if rep == 0:
            """
            Time the first run.
            """
            linear_regression_time = time.time() - linear_regression_time

    mse_train_cv /= n_repetitions
    mse_test_cv /= n_repetitions
    r_score_train_cv /= n_repetitions
    r_score_test_cv /= n_repetitions

    mse_train_boot /= n_repetitions
    mse_test_boot /= n_repetitions
    r_score_train_boot /= n_repetitions
    r_score_test_boot /= n_repetitions

    mse_train_boot_ridge /= n_repetitions
    mse_test_boot_ridge /= n_repetitions
    r_score_train_boot_ridge /= n_repetitions
    r_score_test_boot_ridge /= n_repetitions

    # Data for the neural network:
    X = np.zeros(shape=(n_data_total, 2))
    for i in range(n_data_total): X[i] = x1[i], x2[i]

    q1 = nn.FFNNRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50, 25, 25),
        n_categories = 1,
        n_epochs = 1000,
        batch_size = 50,
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.linear,
        cost_function_derivative = af.mse_derivative,
        verbose = True,
        debug = False,
        scaling = True)

    n_repetitions = 10   # Average to smooth the data.

    mse_train_nn = 0
    mse_test_nn = 0
    r_train_nn = 0
    r_test_nn = 0

    neural_network_time = time.time()
    for rep in range(n_repetitions):
        """
        Repeat and average the data.  Neural network regression.
        """
        print(f"\nrepetition {rep+1} of {n_repetitions}")
        q1.train_neural_network(learning_rate=a_good_learning_rate)
        q1.score()
        mse_train_nn += q1.mse_train
        mse_test_nn += q1.mse_test
        r_train_nn += q1.r_train
        r_test_nn += q1.r_test

        if rep == 0:
            """
            Time the first run.
            """
            neural_network_time = time.time() - neural_network_time

    mse_train_nn /= n_repetitions
    mse_test_nn /= n_repetitions
    r_train_nn /= n_repetitions
    r_test_nn /= n_repetitions

    print("=======================================================================================================")
    print("Linear regression with cross validation (ols) DONT USE THESE VALUES, SOMETHING WRONG WITH R SCORE:")
    print("-----------------------------------------------------------")
    print(f"MSE train: {mse_train_cv}")
    print(f"MSE test: {mse_test_cv}")
    print(f"R train: {r_score_train_cv}")
    print(f"R test: {r_score_test_cv}")
    print("=======================================================================================================")

    print("Linear regression with bootstrapping (ols):")
    print("-----------------------------------------------------------")
    print(f"MSE train: {mse_train_boot}")
    print(f"MSE test: {mse_test_boot}")
    print(f"R train: {r_score_train_boot}")
    print(f"R test: {r_score_test_boot}")
    print("=======================================================================================================")

    print(f"Linear regression with bootstrapping ({ridge_parameter=}):")
    print(f"{linear_regression_time=:.4f} s")
    print("-----------------------------------------------------------")
    print(f"MSE train: {mse_train_boot_ridge}")
    print(f"MSE test: {mse_test_boot_ridge}")
    print(f"R train: {r_score_train_boot_ridge}")
    print(f"R test: {r_score_test_boot_ridge}")
    print("=======================================================================================================")

    print("Neural network regression:")
    print(f"{q1.hidden_layer_sizes=}")
    print(f"{q1.n_epochs=} {q1.batch_size=}")
    print(f"{neural_network_time=:.4f} s")
    print("-----------------------------------------------------------")
    print(f"MSE train: {mse_train_nn}")
    print(f"MSE test: {mse_test_nn}")
    print(f"R train: {r_train_nn}")
    print(f"R test: {r_test_nn}")
    print("=======================================================================================================")

    print("Comparison (nn=neural network, lrb=linear regression bootstrapping):")
    print("-----------------------------------------------------------")
    print(f"MSE train nn/lrb: {mse_train_nn/mse_train_boot} <--- lrb is this times better than nn")
    print(f"MSE test nn/lrb: {mse_test_nn/mse_test_boot} <--- lrb is this times better than nn")
    print(f"R train nn/lrb: {(1 - r_train_nn)/(1 - r_score_train_boot)} <--- lrb is this times closer to 1 than nn")
    print(f"R test nn/lrb: {(1 - r_test_nn)/(1 - r_score_test_boot)} <--- lrb is this times closer to 1 than nn")
    print("=======================================================================================================")


if __name__ == "__main__":
    # classification()
    # regression_vary_learning_rate()
    # regression_vary_regularization_parameters()
    # regression_vary_learning_rate_and_regularization_parameter()
    regression_compare_neural_network_and_ols_ridge()
    pass
