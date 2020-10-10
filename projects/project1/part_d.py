import time
import numpy as np
import matplotlib.pyplot as plt
from common import Regression

def bootstrap():
    """
    Produce contour plots for bias-variance tradeoff as a function of
    ridge regression penalty parameter lambda and polynomial degree.
    Use bootstrapping as resampling technique and ridge regression.
    """
    n_data_points = 400
    max_poly_degree = 10
    noise_factor = 0.2
    n_bootstraps = 50
    repetitions = 10 # Redo the experiment and average the data.

    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)

    n_lambdas = 30
    lambdas = np.logspace(-8, -2, n_lambdas)

    mse_boot = np.zeros((n_lambdas, n_degrees))
    variance_boot = np.zeros((n_lambdas, n_degrees))
    bias_boot = np.zeros((n_lambdas, n_degrees))

    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree,
            split_scale=True)
        
        for j in range(n_lambdas):
            """
            Loop over ridge regression penalty parameters, lambda.
            """
            print(f"lambda {j+1} of {n_lambdas}")
            
            for k in range(n_degrees):
                """
                Loop over degrees.
                """
                mse_boot_tmp, variance_boot_tmp, bias_boot_tmp = \
                    q.bootstrap(degrees[k], n_bootstraps, lambd=lambdas[j])
                
                mse_boot[j, k] += mse_boot_tmp
                variance_boot[j, k] += variance_boot_tmp
                bias_boot[j, k] += bias_boot_tmp

    
    mse_boot /= repetitions
    variance_boot /= repetitions
    bias_boot /= repetitions
    
    X, Y = np.meshgrid(degrees, lambdas)

    fig1, ax0 = plt.subplots(nrows=2, ncols=1, figsize=(8, 9))
    fig1.text(x=0.87, y=0.44, s=r"$log_{10}(error)$", rotation="90", fontsize=15)
    fig1.text(x=0.02, y=0.52, s=r"$\lambda$", rotation="90", fontsize=15)

    ax0 = ax0.ravel()
    mappable = ax0[0].contourf(X, Y, np.log10(variance_boot))
    ax0[0].set_title("Variance", fontsize=17)
    ax0[0].tick_params(labelsize=12)
    cbar = plt.colorbar(mappable, ax=ax0[0], format=r"%.1f")
    cbar.ax.tick_params(labelsize=12)

    mappable = ax0[1].contourf(X, Y, np.log10(bias_boot))
    ax0[1].set_title("Bias", fontsize=17)
    ax0[1].set_xlabel("Polynomial degree", fontsize=15)
    ax0[1].tick_params(labelsize=12)
    cbar = plt.colorbar(mappable, format=r"%.1f")
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(dpi=300, fname="part_d_ridge_bootstrap_variance_bias.png")
    plt.show()


def cross_validation():
    """
    Produce contour plot of MSE as a function of lambda and polynomial
    degree.  Use cross validation resampling and ridge regression.
    """
    n_data_points = 800
    max_poly_degree = 10
    noise_factor = 0.2
    folds = 5
    repetitions = 10 # Redo the experiment and average the data.

    degrees = np.arange(1, max_poly_degree+1, 1)    # Polynomial degrees.
    n_degrees = len(degrees)

    n_lambdas = 100
    lambdas = np.logspace(-12, -4, n_lambdas)   # Ridgre regression penalty parameters.
    
    mse_cv = np.zeros((n_lambdas, n_degrees))
    
    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree,
            split_scale=False)
        
        for j in range(n_lambdas):
            """
            Loop over the different number of ridge regression penalty
            parameters, lambda.
            """
            print(f"lambda {j+1} of {n_lambdas}")
            
            for k in range(n_degrees):
                """
                Loop over the different polynomial degrees.
                """
                mse_cv_tmp, _ = \
                    q.cross_validation(degrees[k], folds, lambd=lambdas[j])
                mse_cv[j, k] += mse_cv_tmp

    mse_cv /= repetitions
    idx = np.unravel_index(np.argmin(mse_cv), mse_cv.shape)
    X, Y = np.meshgrid(degrees, lambdas)

    fig, ax = plt.subplots(figsize=(9, 7))

    mappable = ax.contourf(X, Y, mse_cv)
    ax.set_xlabel("Polynomial degree", fontsize=15)
    ax.set_ylabel(r"$\lambda$",fontsize=15)
    ax.set_title(f"min(MSE) = {np.amin(mse_cv):.4e}: $\lambda$={lambdas[idx[0]]:.4e}, degree={degrees[idx[1]]}", fontsize=15)
    ax.plot(degrees[idx[1]], lambdas[idx[0]], "ro")
    ax.tick_params(labelsize=12)
    cbar = plt.colorbar(mappable, format=r"%.3f")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r"MSE", fontsize=15)
    # plt.savefig(dpi=300, fname="part_d_ridge_cv_mse.png")
    plt.show()


if __name__ == "__main__":
    bootstrap()
    # cross_validation()
    pass