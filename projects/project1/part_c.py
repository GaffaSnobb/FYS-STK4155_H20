import numpy as np
import matplotlib.pyplot as plt
from common import Regression

if __name__ == "__main__":
    """
    Produce the plots for task c.  Plot MSE as a function of polynomial
    degree for different amounts of data points and diffent amounts of
    folds.
    """
    n_data_points_range = np.array([800, 1000, 2000])
    n_data_points = len(n_data_points_range)
    folds = np.array([5, 10])
    n_folds = len(folds)
    n_bootstraps = 50

    max_poly_degree = 10
    noise_factor = 0.2
    repetitions = 10 # Redo the experiment and average the data.
    
    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)
    
    fig, axs = plt.subplots(nrows=n_data_points, ncols=n_folds,
        figsize=(9, 9))
    fig.tight_layout(pad=3)
    fig.text(x=0, y=0.48, s="MSE", fontsize=15, rotation="vertical")
    fig.text(x=0.4, y=0.02, s="Polynomial degree", fontsize=15)

    for k in range(n_data_points):
        """
        Loop over the different amounts of data ponts.
        """
        print(f"data points {k+1} of {n_data_points}, {n_data_points_range[k]}")

        mse_cv = np.zeros(n_degrees)
        mse_boot = np.zeros(n_degrees)
        for l in range(n_folds):
            """
            Loop over the different folds.
            """
            for i in range(repetitions):
                """
                Repeat the experiment and average the produced values.
                """
                print(f"repetition {i+1} of {repetitions}, fold {l+1} of {n_folds}")
                q = Regression(n_data_points_range[k], noise_factor,
                    max_poly_degree, split_scale=True)
                
                for j in range(n_degrees):
                    """
                    Loop over polynomial degrees.
                    """
                    mse_cv_tmp, _ = q.cross_validation(degree=degrees[j],
                        folds=folds[l])
                    mse_boot_tmp, _, _ = q.bootstrap(degree=degrees[j],
                        n_bootstraps=n_bootstraps)
                    mse_cv[j] += mse_cv_tmp
                    mse_boot[j] += mse_boot_tmp

            mse_cv /= repetitions
            mse_boot /= repetitions

            axs[k, l].plot(degrees, mse_cv, label="Cross validation",
                color="black")
            axs[k, l].plot(degrees, mse_boot, label="Bootstrap", color="grey",
                linestyle="dashed")
            axs[k, l].tick_params(labelsize=12)
            axs[k, l].set_title(f"Data points = {n_data_points_range[k]}, folds = {folds[l]}")
            axs[k, l].set_ylim(0.04, 0.07)

    axs[0, 1].set_yticklabels(labels=[])
    axs[1, 1].set_yticklabels(labels=[])
    axs[2, 1].set_yticklabels(labels=[])
    axs[0, 0].set_xticklabels(labels=[])
    axs[0, 1].set_xticklabels(labels=[])
    axs[1, 0].set_xticklabels(labels=[])
    axs[1, 1].set_xticklabels(labels=[])
    axs[0, 1].legend(fontsize=12)
    # fig.savefig(dpi=300, fname="part_c_comparing_bootstrap_cv.png")
    plt.show()


