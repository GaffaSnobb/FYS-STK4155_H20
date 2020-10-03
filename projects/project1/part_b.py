import numpy as np
import matplotlib.pyplot as plt
from common import Regression

if __name__ == "__main__":
    n_data_points = 400
    max_poly_degree = 10
    noise_factor = 0.1
    n_bootstraps = 50
    repetitions = 10 # Redo the experiment and average the data.
    
    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)
    
    mse = np.zeros(n_degrees)
    bias = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)
    
    
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
            mse_tmp, variance_tmp, bias_tmp = q.bootstrap(degree=degrees[j], n_bootstraps=n_bootstraps)
            mse[j] += mse_tmp
            variance[j] += variance_tmp
            bias[j] += bias_tmp

    mse /= repetitions
    variance /= repetitions
    bias /= repetitions

    plt.plot(degrees, mse, label="mse", color="black")
    plt.plot(degrees, variance, label="variance", color="grey", linestyle="dashed")
    plt.plot(degrees, bias, label="bias", color="black", linestyle="dotted")
    plt.title("Bootstrap", fontsize=17)
    plt.xlabel("Error", fontsize=15)
    plt.ylabel("Polynomial degree", fontsize=15)
    plt.tick_params(labelsize=12)
    plt.legend(fontsize=17)
    plt.show()