import time
import numpy as np
import matplotlib.pyplot as plt
from common import Regression

def bootstrap():
    n_data_points = 400
    max_poly_degree = 15
    noise_factor = 0.2
    n_bootstraps = 50
    repetitions = 1 # Redo the experiment and average the data.

    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)

    n_lambdas = 20
    # lambdas = np.linspace(-2, 1, n_lambdas)
    lambdas = np.logspace(-8, -2, n_lambdas)

    mse_boot = np.zeros((n_lambdas, n_degrees))
    
    bootstrap_time = 0

    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree, split_scale=True)
        
        for j in range(n_lambdas):
            """
            Loop over lambdas.
            """
            print(f"lambda {j+1} of {n_lambdas}")
            for k in range(n_degrees):
                """
                Loop over degrees.
                """
                bootstrap_time_tmp = time.time()
                mse_boot_tmp, _, _ = \
                    q.bootstrap(degrees[k], n_bootstraps, lambd=lambdas[j])
                bootstrap_time += time.time() - bootstrap_time_tmp
                mse_boot[j, k] += mse_boot_tmp

    bootstrap_time /= n_lambdas*n_degrees
    print(f"avg. bootstrap time: {bootstrap_time}")
    
    
    mse_boot /= repetitions
    idx = np.unravel_index(np.argmin(mse_boot), mse_boot.shape)
    X, Y = np.meshgrid(degrees, lambdas)
    plt.contourf(X, Y, np.log10(mse_boot))
    plt.xlabel("degrees")
    plt.ylabel("lambdas")
    plt.title(f"min: lambda={lambdas[idx[0]]}, degree={degrees[idx[1]]}")
    plt.colorbar()
    plt.show()


def cross_validation():
    n_data_points = 400
    max_poly_degree = 10
    noise_factor = 0.2
    folds = 5
    repetitions = 10 # Redo the experiment and average the data.

    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)

    n_lambdas = 40
    lambdas = np.logspace(-8, -2, n_lambdas)
    # lambdas = np.linspace(-1, 1, n_lambdas)
    mse_cv = np.zeros((n_lambdas, n_degrees))
    
    cv_time = 0

    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree,
            split_scale=False)
        
        for j in range(n_lambdas):
            print(f"lambda {j+1} of {n_lambdas}")
            for k in range(n_degrees):

                cv_time_tmp = time.time()
                mse_cv_tmp = \
                    q.cross_validation(degrees[k], folds, lambd=lambdas[j])
                cv_time += time.time() - cv_time_tmp
                mse_cv[j, k] += mse_cv_tmp

    cv_time /= n_lambdas*n_degrees*repetitions
    print(f"avg. cv time: {cv_time}")
    mse_cv /= repetitions
    idx = np.unravel_index(np.argmin(mse_cv), mse_cv.shape)
    X, Y = np.meshgrid(degrees, lambdas)
    plt.contourf(X, Y, np.log10(mse_cv))
    plt.xlabel("degrees")
    plt.ylabel("lambdas")
    plt.title(f"min: lambda={lambdas[idx[0]]}, degree={degrees[idx[1]]}")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # bootstrap()
    cross_validation()
    pass