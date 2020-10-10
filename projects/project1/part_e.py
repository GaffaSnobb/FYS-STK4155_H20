import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numpy.core.numeric import cross
from common import Regression
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def contour():
    n_data_points = 400
    max_poly_degree = 8
    noise_factor = 0.2
    folds = 5
    repetitions = 20 # Redo the experiment and average the data.

    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)

    n_alphas = 10
    # alphas = np.logspace(-4, -1, n_alphas)
    alphas = np.logspace(-12, -6, n_alphas)
    mse_cv = np.zeros((n_alphas, n_degrees))
    total_time = 0
    
    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        rep_time = time.time()
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree,
            split_scale=False)
        
        for j in range(n_alphas):
            print(f"alpha {j+1} of {n_alphas}, alpha = {alphas[j]}")
            for k in range(n_degrees):

                mse_cv_tmp, _ = \
                    q.cross_validation(degrees[k], folds, alpha=alphas[j])
                mse_cv[j, k] += mse_cv_tmp

        rep_time = time.time() - rep_time
        total_time += rep_time
        print(f"repetition {i+1} took: {rep_time:.2f}s")
        print(f"cumulative time: {total_time:.2f}s")

    print(f"total time {total_time:.2f}s")

    mse_cv /= repetitions

    X, Y = np.meshgrid(degrees, alphas)

    idx = np.unravel_index(np.argmin(mse_cv), mse_cv.shape)
        
    fig, ax = plt.subplots(figsize=(9, 7))
    mappable = ax.contourf(X, Y, mse_cv)
    ax.set_xlabel("Polynomial degree", fontsize=15)
    ax.set_ylabel(r"$\lambda$", fontsize=15)
    ax.set_title(f"min(MSE) = {np.amin(mse_cv):.4e} \nat $\lambda$={alphas[idx[0]]:.4e}, degree={degrees[idx[1]]}", fontsize=15)
    ax.tick_params(labelsize=12)
    cbar = plt.colorbar(mappable)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r"MSE", fontsize=15)
    plt.savefig(dpi=300,
        fname=f"part_e_lasso_{folds}foldcv_lambda_polydeg_mse_{n_data_points}dpoints_{repetitions}reps.png")

    return degrees[idx[1]]


@ignore_warnings(category=ConvergenceWarning)
def cross_validation(best_degree):
    n_data_points = 400
    max_poly_degree = best_degree
    noise_factor = 0.2
    folds = 5
    repetitions = 20 # Redo the experiment and average the data.

    n_alphas = 40
    alphas = np.logspace(-12, -6, n_alphas)
    mse_cv = np.zeros(n_alphas)
    total_time = 0
    
    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        rep_time = time.time()
        print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree,
            split_scale=False)
        
        for j in range(n_alphas):
            print(f"alpha {j+1} of {n_alphas}, alpha = {alphas[j]}")

            mse_cv_tmp, _ = \
                q.cross_validation(best_degree, folds, alpha=alphas[j])
            mse_cv[j] += mse_cv_tmp

        rep_time = time.time() - rep_time
        total_time += rep_time
        print(f"repetition {i+1} took: {rep_time:.2f}s")
        print(f"cumulative time: {total_time:.2f}s")

    print(f"total time {total_time:.2f}s")

    mse_cv /= repetitions

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.semilogx(alphas, mse_cv)
    ax.set_title(f"min(MSE) = {np.amin(mse_cv):.4e} \npolynomial degree = {best_degree}", fontsize=17)
    ax.set_xlabel(r"$\lambda$", fontsize=15)
    ax.set_ylabel("MSE", fontsize=15)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    ax.tick_params(labelsize=12)
    plt.savefig(dpi=300,
        fname=f"part_e_lasso_{folds}foldcv_lambda_mse_{n_data_points}dpoints_{repetitions}reps.png")

@ignore_warnings(category=ConvergenceWarning)
def bootstrap():
    n_data_points = 400
    max_poly_degree = 7
    noise_factor = 0.2
    n_bootstraps = 50
    repetitions = 10 # Redo the experiment and average the data.

    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)

    n_alphas = 30
    # alphas = np.linspace(-2, 1, n_alphas)
    alphas = np.logspace(-8, -5, n_alphas)

    mse_boot = np.zeros((n_alphas, n_degrees))
    variance_boot = np.zeros((n_alphas, n_degrees))
    bias_boot = np.zeros((n_alphas, n_degrees))
    total_time = 0

    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        rep_time = time.time()
        cum_alpha_time = 0
        # print(f"repetition {i+1} of {repetitions}")
        q = Regression(n_data_points, noise_factor, max_poly_degree, split_scale=True)
        
        for j in range(n_alphas):
            """
            Loop over alphas.
            """
            alpha_time = time.time()
            # print(f"alpha {j+1} of {n_alphas}")
            for k in range(n_degrees):
                """
                Loop over degrees.
                """
                mse_boot_tmp, variance_boot_tmp, bias_boot_tmp = \
                    q.bootstrap(degrees[k], n_bootstraps, alpha=alphas[j])
                
                mse_boot[j, k] += mse_boot_tmp
                variance_boot[j, k] += variance_boot_tmp
                bias_boot[j, k] += bias_boot_tmp

            alpha_time = time.time() - alpha_time
            cum_alpha_time += alpha_time
            print(f"alpha {j+1} of {n_alphas} took: {alpha_time:.2f}s")
            print(f"cumulative time for alpha {j+1}: {cum_alpha_time:.2f}s\n")

    
        rep_time = time.time() - rep_time
        total_time += rep_time
        print(f"repetition {i+1} of {repetitions} took: {rep_time:.2f}s")
        print(f"cumulative time for repetition {repetitions}: {total_time:.2f}s\n")

    print(f"total time {total_time:.2f}s")

    mse_boot /= repetitions
    variance_boot /= repetitions
    bias_boot /= repetitions
    
    X, Y = np.meshgrid(degrees, alphas)
    
    fig0, ax0 = plt.subplots(figsize=(10, 8))
    idx = np.unravel_index(np.argmin(mse_boot), mse_boot.shape)
    mappable = ax0.contourf(X, Y, (mse_boot))
    ax0.set_xlabel("Polynomial degree", fontsize=15)
    ax0.set_ylabel(r"$\lambda$",fontsize=15)
    ax0.set_title(f"min(MSE) = {np.amin(mse_boot):.4e} \nat lambda = {alphas[idx[0]]:.4e}, degree = {degrees[idx[1]]}", fontsize=17)
    ax0.tick_params(labelsize=12)
    cbar = plt.colorbar(mappable)
    cbar.set_label(r"MSE", fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    fig0.savefig(dpi=300,
        fname=f"part_e_{n_bootstraps}boots_lambda_poly_mse_{n_data_points}dpoints_{repetitions}reps.png")

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    mappable = ax1.contourf(X, Y, (variance_boot))
    ax1.set_xlabel("Polynomial degree", fontsize=15)
    ax1.set_ylabel(r"$\lambda$",fontsize=15)
    ax1.tick_params(labelsize=12)
    cbar = plt.colorbar(mappable)
    cbar.set_label(r"Var", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    fig1.savefig(dpi=300,
        fname=f"part_e_{n_bootstraps}boots_lambda_poly_var_{n_data_points}dpoints_{repetitions}reps.png")

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    mappable = ax2.contourf(X, Y, (bias_boot))
    ax2.set_xlabel("Polynomial degree", fontsize=15)
    ax2.set_ylabel(r"$\lambda$",fontsize=15)
    ax2.tick_params(labelsize=12)
    cbar = plt.colorbar(mappable)
    cbar.set_label(r"Bias", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    fig2.savefig(dpi=300,
        fname=f"part_e_{n_bootstraps}boots_lambda_poly_bias_{n_data_points}dpoints_{repetitions}reps.png")



if __name__ == "__main__":
    # best_degree = contour()
    # cross_validation(best_degree)
    bootstrap()