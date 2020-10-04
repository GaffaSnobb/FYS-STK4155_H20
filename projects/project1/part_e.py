import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import cross
from common import Regression


def contour():
    n_data_points = 400
    max_poly_degree = 10
    noise_factor = 0.2
    folds = 5
    repetitions = 1 # Redo the experiment and average the data.

    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)

    n_alphas = 20
    # alphas = np.linspace(0, 1, n_alphas)
    alphas = np.logspace(-8, -2, n_alphas)
    mse_cv = np.zeros((n_alphas, n_degrees))
    
    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree,
            split_scale=False)
        
        for j in range(n_alphas):
            print(f"alpha {j+1} of {n_alphas}, alpha = {alphas[j]}")
            for k in range(n_degrees):

                mse_cv_tmp, _ = \
                    q.cross_validation(degrees[k], folds, alpha=alphas[j])
                mse_cv[j, k] += mse_cv_tmp

    mse_cv /= repetitions

    X, Y = np.meshgrid(degrees, alphas)

    idx = np.unravel_index(np.argmin(mse_cv), mse_cv.shape)
    plt.contourf(X, Y, mse_cv)
    plt.xlabel("Degrees", fontsize=15)
    plt.ylabel("Lambdas", fontsize=15)
    plt.title(f"min: lambda={alphas[idx[0]]}, degree={degrees[idx[1]]}", fontsize=15)
    plt.tick_params(labelsize=12)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    plt.show()


def cross_validation():
    n_data_points = 400
    max_poly_degree = 7
    noise_factor = 0.2
    folds = 5
    repetitions = 1 # Redo the experiment and average the data.

    n_alphas = 40
    # alphas = np.linspace(1e-18, 1e-1, n_alphas)
    alphas = np.logspace(-10, -1, n_alphas)
    mse_cv = np.zeros(n_alphas)
    
    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree,
            split_scale=False)
        
        for j in range(n_alphas):
            print(f"alpha {j+1} of {n_alphas}, alpha = {alphas[j]}")

            mse_cv_tmp, _ = \
                q.cross_validation(max_poly_degree, folds, alpha=alphas[j])
            mse_cv[j] += mse_cv_tmp

    mse_cv /= repetitions


    plt.semilogx(alphas, mse_cv)
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.show()


def bootstrap():
    n_data_points = 400
    max_poly_degree = 10
    noise_factor = 0.2
    n_bootstraps = 50
    repetitions = 1 # Redo the experiment and average the data.

    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)

    n_alphas = 20
    # alphas = np.linspace(-2, 1, n_alphas)
    alphas = np.logspace(-4, 0, n_alphas)

    mse_boot = np.zeros((n_alphas, n_degrees))
    variance_boot = np.zeros((n_alphas, n_degrees))
    bias_boot = np.zeros((n_alphas, n_degrees))

    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree, split_scale=True)
        
        for j in range(n_alphas):
            """
            Loop over alphas.
            """
            print(f"alpha {j+1} of {n_alphas}")
            for k in range(n_degrees):
                """
                Loop over degrees.
                """
                mse_boot_tmp, variance_boot_tmp, bias_boot_tmp = \
                    q.bootstrap(degrees[k], n_bootstraps, alpha=alphas[j])
                
                mse_boot[j, k] += mse_boot_tmp
                variance_boot[j, k] += variance_boot_tmp
                bias_boot[j, k] += bias_boot_tmp

    
    
    mse_boot /= repetitions
    variance_boot /= repetitions
    bias_boot /= repetitions
    
    X, Y = np.meshgrid(degrees, alphas)
    
    fig0, ax0 = plt.subplots()
    idx = np.unravel_index(np.argmin(mse_boot), mse_boot.shape)
    mappable = ax0.contourf(X, Y, np.log10(mse_boot))
    ax0.set_xlabel("degrees")
    ax0.set_ylabel("alphas")
    ax0.set_title(f"mse\nmin: alpha={alphas[idx[0]]}, degree={degrees[idx[1]]}")
    cbar = plt.colorbar(mappable)
    # plt.show()

    fig1, ax1 = plt.subplots()
    mappable = ax1.contourf(X, Y, np.log10(variance_boot))
    ax1.set_title("variance")
    ax1.set_xlabel("degrees")
    ax1.set_ylabel("alphas")
    cbar = plt.colorbar(mappable)
    # cbar.set_label(r"$log_{10}$ error", fontsize=40)
    # cbar.ax.tick_params(labelsize=30)
    # plt.show()

    fig2, ax2 = plt.subplots()
    mappable = ax2.contourf(X, Y, np.log10(bias_boot))
    ax2.set_title("bias")
    ax2.set_xlabel("degrees")
    ax2.set_ylabel("alphas")
    cbar = plt.colorbar(mappable)
    # cbar.set_label(r"$log_{10}$ error", fontsize=40)
    # cbar.ax.tick_params(labelsize=30)
    plt.show()

if __name__ == "__main__":
    contour()
    # cross_validation()
    # bootstrap()