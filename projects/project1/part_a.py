import numpy as np
import matplotlib.pyplot as plt
from common import Regression, features

if __name__ == "__main__":
    """
    Produce the plots for task 1a.  Produce plots of MSE as a function
    of polynomial degree.  Produce the plots of R squared as a function
    of polynomial degree.  Produce the plot for confidence itervals.
    """
    n_data_points_range = np.array([800, 1600, 5000, 10000])
    n_data_points = len(n_data_points_range)
    max_poly_degree = 15
    noise_factor = 0.2
    repetitions = 20    # Redo the experiment and average the data.
    
    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)
    
    fig0, ax0 = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    ax0 = ax0.ravel()

    fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    ax1 = ax1.ravel()

    fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    ax2 = ax2.ravel()
    
    for k in range(n_data_points):
        """
        Loop over different number of data points.
        """
        print(f"data point set {k+1} of {n_data_points}")
        mse_train = np.zeros(n_degrees)
        mse_test = np.zeros(n_degrees)
        r_score_train = np.zeros(n_degrees)
        r_score_test = np.zeros(n_degrees)
        
        for i in range(repetitions):
            """
            Repeat the experiment and average the produced values.
            """
            print(f"repetition {i+1} of {repetitions}")
            q = Regression(n_data_points_range[k], noise_factor,
                max_poly_degree)
            
            for j in range(n_degrees):
                """
                Loop over polynomial degrees.
                """
                r_score_train_tmp, mse_train_tmp, r_score_test_tmp, \
                    mse_test_tmp, beta_tmp, var_beta_tmp = \
                    q.standard_least_squares_regression(degree=j)

                r_score_train[j] += r_score_train_tmp
                r_score_test[j] += r_score_test_tmp
                mse_train[j] += mse_train_tmp
                mse_test[j] += mse_test_tmp
                
                if j == 5:
                    """
                    Fetch the 5th degree polynomial data for confidence
                    interval.
                    """
                    beta5 = beta_tmp
                    var_beta5 = var_beta_tmp

        r_score_train /= repetitions
        r_score_test /= repetitions
        mse_train /= repetitions
        mse_test /= repetitions

        ax0[k].plot(degrees, mse_train, label="train", color="grey",
            linestyle="dashed")
        ax0[k].plot(degrees, mse_test, label="test", color="black")
        ax0[k].legend()
        ax0[k].set_title(f"Data points: {n_data_points_range[k]}")

        ax1[k].plot(degrees, r_score_train, label="train", color="grey",
            linestyle="dashed")
        ax1[k].plot(degrees, r_score_test, label="test", color="black")
        ax1[k].legend()
        ax1[k].set_title(f"Data points: {n_data_points_range[k]}")

        ax2[k].errorbar(np.arange(0, features(5), 1), beta5, var_beta5,
            fmt='o')
        ax2[k].set_title('Confidence intervals of '+r'$\beta$')
    
    ax0[0].set_ylabel("MSE")
    ax0[2].set_ylabel("MSE")
    ax0[2].set_xlabel("Polynomial degree")
    ax0[3].set_xlabel("Polynomial degree")

    ax1[0].set_ylabel(r"$R^2$")
    ax1[2].set_ylabel(r"$R^2$")
    ax1[2].set_xlabel("Polynomial degree")
    ax1[3].set_xlabel("Polynomial degree")

    ax2[0].set_ylabel(r'$\beta$'+'-value')
    ax2[2].set_ylabel(r'$\beta$'+'-value')
    ax2[2].set_xlabel(r'$\beta$'+'-number')
    ax2[3].set_xlabel(r'$\beta$'+'-number')
    
    plt.show()