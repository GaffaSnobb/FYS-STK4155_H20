import numpy as np
import matplotlib.pyplot as plt
from common import Regression

if __name__ == "__main__":
    n_data_points = np.array([800, 1000, 2000])
    max_poly_degree = 10
    noise_factor = 0.2
    n_bootstraps = 50
    folds = np.array([5,10])
    repetitions = 10 # Redo the experiment and average the data.
    
    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)
    mse_cv = np.zeros(n_degrees)
    mse_boot = np.zeros(n_degrees)
    
    fig, axs = plt.subplots(len(n_data_points),len(folds))
   

    for k in range(len(n_data_points)):
        for l in range(len(folds)):
            for i in range(repetitions):
                """
                Repeat the experiment and average the produced values.
                """
                print(f"repetition {i+1} of {repetitions}")
                q = Regression(n_data_points[k], noise_factor, max_poly_degree, split_scale=True)
                for j in range(n_degrees):
                    """
                    Loop over polynomial degrees.
                    """
                    mse_cv_tmp, _ = q.cross_validation(degree=degrees[j], folds=folds[l])
                    mse_boot_tmp, _, _ = q.bootstrap(degree=degrees[j], n_bootstraps=n_bootstraps)
                    mse_cv[j] += mse_cv_tmp
                    mse_boot[j] += mse_boot_tmp

            mse_cv /= repetitions
            mse_boot /= repetitions

            axs[k,l].plot(degrees, mse_cv, label="Cross validation", color="black")
            axs[k,l].plot(degrees, mse_boot, label="Bootstrap", color="grey", linestyle="dashed")
            #plt.xlabel("Polynomial degree", fontsize=15)
            #plt.ylabel("MSE", fontsize=15)
            #plt.title("Data points %d" %n_data_points)
            #plt.tick_params(labelsize=12)
            axs[k,l].set_title("Data points = %d, folds = %d" %(n_data_points[k],folds[l]))
            axs[k,l].set(xlabel="Polynomial degree", ylabel="MSE")
            axs[k,l].label_outer()
    plt.legend()
    plt.show()


