import numpy as np
import matplotlib.pyplot as plt
from common import Regression

if __name__ == "__main__":
    n_data_points = 800
    max_poly_degree = 10
    noise_factor = 0.2
    n_bootstraps = 50
    folds = 5
    repetitions = 10 # Redo the experiment and average the data.
    
    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)
    mse_cv = np.zeros(n_degrees)
    mse_boot = np.zeros(n_degrees)
    

    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree, split_scale=True)
        for j in range(n_degrees):
            """
            Loop over polynomial degrees.
            """
            mse_cv_tmp = q.cross_validation(degree=degrees[j], folds=folds)
            mse_boot_tmp, _, _ = q.bootstrap(degree=degrees[j], n_bootstraps=n_bootstraps)
            mse_cv[j] += mse_cv_tmp
            mse_boot[j] += mse_boot_tmp

    mse_cv /= repetitions
    mse_boot /= repetitions

    plt.plot(degrees, mse_cv, label="mse cv")
    plt.plot(degrees, mse_boot, label="mse boot")
    plt.legend()
    plt.show()


