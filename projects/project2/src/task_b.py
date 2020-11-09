import numpy as np
import matplotlib.pyplot as plt
import common


def temporary_sklearn_solution():
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    n_data_total = 400
    poly_degree = 10

    x1 = np.random.uniform(0, 1, n_data_total)
    x2 = np.random.uniform(0, 1, n_data_total)
    X = common.create_design_matrix_two_dependent_variables(x1,
        x2, n_data_total, poly_degree)
    y = common.franke_function(x1, x2)

    N = 50
    # learning_rates = np.logspace(-3, -1, N)
    learning_rates = np.linspace(1e-5, 1e-3, N)
    mse_train = np.zeros(N)
    mse_test = np.zeros(N)

    for i in range(N):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, shuffle=True)
        
        reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000,
            tol=1e-3, learning_rate="constant", eta0=learning_rates[i]))
        reg.fit(X_train, y_train)
        
        mse_test[i] = common.mean_squared_error(reg.predict(X_test), y_test)
        mse_train[i] = common.mean_squared_error(reg.predict(X_train), y_train)

    plt.plot(learning_rates, mse_train, label="train")
    plt.plot(learning_rates, mse_test, label="test")
    plt.xlabel("learning rate")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
    


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
    # neural_network_franke()
    temporary_sklearn_solution()

    # q1 = common.FFNN(X=digits.images, y=digits.target,
    #     hidden_layer_sizes=(50, 20, 20), n_categories=10,
    #     hidden_layer_activation_function=common.sigmoid,
    #     output_activation_function=common.softmax,
    #     cost_function=common.cross_entropy_derivative,
    #     verbose=True)
    pass