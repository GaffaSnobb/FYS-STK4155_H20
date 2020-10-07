import numpy as np
import matplotlib.pyplot as plt
from common import Regression

if __name__ == "__main__":
    n_data_points = np.array([400, 800, 2000,10000])
    max_poly_degree = 10
    noise_factor = 0.1
    n_bootstraps = 50
    repetitions = 100 # Redo the experiment and average the data.
    
    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)
    
    mse = np.zeros(n_degrees)
    bias = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    # fig.suptitle("Bootstrap", fontsize=17)
    fig.tight_layout(pad=3)
    fig.text(x=0.4, y=0.02, s="Polynomial degree", fontsize=15)
    fig.text(x=0.01, y=0.48, s="Error", fontsize=15, rotation="vertical")
    axs = axs.ravel()
    
    for k in range(len(n_data_points)):
        print(f"data points {k+1} of {len(n_data_points)}, {n_data_points[k]}")
        for i in range(repetitions):
            """
            Repeat the experiment and average the produced values.
            """
            print(f"repetition {i+1} of {repetitions}")
            q = Regression(n_data_points[k], noise_factor, max_poly_degree)
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

        axs[k].plot(degrees, mse, label="mse", color="black")
        axs[k].plot(degrees, variance, label="variance", color="grey", linestyle="dashed")
        axs[k].plot(degrees, bias, label="bias", color="black", linestyle="dotted")
        axs[k].set_title("Data points %d" %n_data_points[k])
        axs[k].tick_params(labelsize=12)
        axs[k].set_ylim(-0.01, 0.07)
        # axs[k].set(xlabel="Polynomial degree", ylabel="Error")
        # axs[k].label_outer()
        #axs[k].tick_params(labelsize=12)
    
    axs[1].set_yticklabels(labels=[])
    axs[3].set_yticklabels(labels=[])
    axs[0].set_xticklabels(labels=[])
    axs[1].set_xticklabels(labels=[])
    axs[1].legend(fontsize=12, loc="upper right")
    fig.savefig(dpi=300, fname="part_b_bootstrap_bias_variance_tradeoff.pdf")
