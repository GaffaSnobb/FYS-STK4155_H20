import numpy as np
import matplotlib.pyplot as plt
from common import Regression

if __name__ == "__main__":
    n_data_points = 800
    max_poly_degree = 20
    noise_factor = 0.2
    folds = 5
    
    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)
    
    mse_train = np.empty(n_degrees)
    mse_test = np.empty(n_degrees)
    r_score_train = np.empty(n_degrees)
    r_score_test = np.empty(n_degrees)
    
    q = Regression(n_data_points, noise_factor, max_poly_degree)
    
    for i in range(n_degrees):
        """
        Loop over polynomial degrees.
        """
        r_score_train[i], mse_train[i], r_score_test[i], mse_test[i] = \
            q.standard_least_squares_regression(degree=i)

    plt.plot(degrees, mse_train, label="train")
    plt.plot(degrees, mse_test, label="test")
    plt.title("mse")
    plt.show()

    plt.plot(degrees, r_score_train, label="train")
    plt.plot(degrees, r_score_test, label="test")
    plt.title("r score")
    plt.show()