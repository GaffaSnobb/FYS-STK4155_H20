import numpy as np
import matplotlib.pyplot as plt
import common


def create_1d_design_matrix(x, n_data_points, poly_degree):
    X = np.empty((n_data_points, poly_degree+1))
    X[:, 0] = 1 # Intercept.

    x = np.sort(x)

    for i in range(1, poly_degree+1):
        X[:, i] = x**i

    return X


def sgd():

    n_data_points = 10  # Number of data points.
    poly_degree = 2
    n_features = common.features(poly_degree)
    n_gradient_iterations = 10
    gradient_step_size = 0.1
    
    x1 = np.random.uniform(0, 1, n_data_points)
    x2 = np.random.uniform(0, 1, n_data_points)

    # X = common.create_design_matrix(x1, x2, n_data_points, poly_degree)
    # y = common.franke_function(x1, x2)
    # beta = np.zeros(n_features)
    X = create_1d_design_matrix(x1, n_data_points, poly_degree)
    y = 4 + 3*x1 + np.random.randn(n_data_points)
    
    beta = np.zeros(poly_degree+1)

    for _ in range(n_gradient_iterations):
        gradient = X.T@(X@beta - y)*2/n_data_points
        beta -= gradient_step_size*gradient

    res = X@beta

    plt.plot(x1, y, "r.")
    plt.plot(np.sort(x1), res)
    plt.show()


if __name__ == "__main__":
    sgd()