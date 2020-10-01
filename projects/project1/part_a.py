import numpy as np
import matplotlib.pyplot as plt
from common import Regression

if __name__ == "__main__":
    n_data_points = 800
    #[800, 1600, 5000, 10000]
    max_poly_degree = 15
    noise_factor = 0.2
    folds = 5
    repetitions = 20    # Redo the experiment and average the data.
    
    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)
    
    mse_train = np.zeros(n_degrees)
    mse_test = np.zeros(n_degrees)
    r_score_train = np.zeros(n_degrees)
    r_score_test = np.zeros(n_degrees)
    
    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree)
        for j in range(n_degrees):
            """
            Loop over polynomial degrees.
            """
            r_score_train_tmp, mse_train_tmp, r_score_test_tmp, mse_test_tmp = \
                q.standard_least_squares_regression(degree=j)

            r_score_train[j] += r_score_train_tmp
            r_score_test[j] += r_score_test_tmp
            mse_train[j] += mse_train_tmp
            mse_test[j] += mse_test_tmp

    r_score_train /= repetitions
    r_score_test /= repetitions
    mse_train /= repetitions
    mse_test /= repetitions

    plt.plot(degrees, mse_train, label="train")
    plt.plot(degrees, mse_test, label="test")
    plt.legend()
    plt.title("mse")
    plt.show()

    plt.plot(degrees, r_score_train, label="train")
    plt.plot(degrees, r_score_test, label="test")
    plt.title("r score")
    plt.legend()
    plt.show()