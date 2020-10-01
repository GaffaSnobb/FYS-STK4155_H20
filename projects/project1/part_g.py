from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from common import Regression, create_design_matrix


class Terrain(Regression):
    def __init__(self, max_poly_degree, every_nth=50):
        """
        Parameters
        ----------
        max_poly_degree : int
            The maximum polynomial degree.

        every_nth : int
            Keep only every nth data point.  This is the 'step' in
            slicing ([start:stop:step]).
        """
        terrain = imread("SRTM_data_Norway_1.tif")
        self.max_poly_degree = max_poly_degree
        
        self.y = terrain[0:terrain.shape[1]] # Equal dimension length.
        self.y = self.y[::every_nth, ::every_nth]   # Keep only a part of the values.
        self.sliced_shape = self.y.shape
        self.y = self.y.ravel()
        self.n_data_points = self.y.shape[0]

        x1 = np.linspace(0, 1, self.n_data_points)
        x2 = np.linspace(0, 1, self.n_data_points)
        self.X = create_design_matrix(x1, x2, self.n_data_points, self.max_poly_degree)

        self._split_scale()


def ridge_cv():
    max_poly_degree = 5
    folds = 5
    repetitions = 10 # Redo the experiment and average the data.

    n_lambdas = 40
    lambdas = np.logspace(-18, 1, n_lambdas)
    mse_cv = np.zeros(n_lambdas)
    mse_cv_training = np.zeros(n_lambdas)
    
    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Terrain(max_poly_degree)
        for j in range(n_lambdas):
            mse_cv_tmp, mse_cv_training_tmp = \
                q.cross_validation(degree=max_poly_degree, folds=folds, lambd=lambdas[j])
            mse_cv[j] += mse_cv_tmp
            mse_cv_training[j] += mse_cv_training_tmp
            

    mse_cv /= repetitions
    mse_cv_training /= repetitions

    print(q.sliced_shape)

    plt.semilogx(lambdas, mse_cv, label="mse cv")
    plt.semilogx(lambdas, mse_cv_training, label="mse cv training")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.title("Ridge regression")
    plt.legend()
    plt.show()


def lasso_cv():
    max_poly_degree = 5
    folds = 5
    repetitions = 5 # Redo the experiment and average the data.

    n_alphas = 40
    alphas = np.logspace(-18, 1, n_alphas)
    mse_cv = np.zeros(n_alphas)
    mse_cv_training = np.zeros(n_alphas)
    
    for i in range(repetitions):
        """
        Repeat the experiment and average the produced values.
        """
        print(f"repetition {i+1} of {repetitions}")
        q = Terrain(max_poly_degree)
        for j in range(n_alphas):
            mse_cv_tmp, mse_cv_training_tmp = \
                q.cross_validation(degree=max_poly_degree, folds=folds, alpha=alphas[j])
            mse_cv[j] += mse_cv_tmp
            mse_cv_training[j] += mse_cv_training_tmp
            

    mse_cv /= repetitions
    mse_cv_training /= repetitions

    print(q.sliced_shape)

    plt.semilogx(alphas, mse_cv, label="mse cv")
    plt.semilogx(alphas, mse_cv_training, label="mse cv training")
    plt.xlabel("alpha")
    plt.ylabel("MSE")
    plt.title("Lasso regression")
    plt.legend()
    plt.show()


def plain_ols():
    """
    Use OLS on the terrain data.  Show the change in overfitting, as MSE
    and R score as a function of polynomial degree, for different number
    of data points.
    """
    max_poly_degree = 20
    repetitions = 100    # Redo the experiment and average the data.
    
    degrees = np.arange(1, max_poly_degree+1, 1)
    n_degrees = len(degrees)
    steps = [125, 100, 75, 50]  # These are slice values for the terrain data array.
    
    fig0, ax0 = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    fig0.tight_layout(pad=2.5)
    fig0.text(x=0.4, y=0.02, s="Polynomial degree", fontsize=15)
    fig0.text(x=0.03, y=0.48, s="MSE", fontsize=15, rotation="vertical")
    ax0 = ax0.ravel()
    
    fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    fig1.tight_layout(pad=2.5)
    fig1.text(x=0.4, y=0.02, s="Polynomial degree", fontsize=15)
    fig1.text(x=0.03, y=0.5, s="$R^2$", fontsize=15, rotation="vertical")
    ax1 = ax1.ravel()

    for k in range(len(steps)):
        """
        'step' defines the slicing step length of the original data.
        Loop over steps to see how the model changes as the number of
        data points changes.
        """
        print(f"step length {steps[k]}")
        mse_train_avg = np.zeros(n_degrees)
        mse_test_avg = np.zeros(n_degrees)
        r_score_train_avg = np.zeros(n_degrees)
        r_score_test_avg = np.zeros(n_degrees)
        
        for i in range(repetitions):
            """
            Repeat the experiment and average the produced values.
            """
            print(f"repetition {i+1} of {repetitions}")
            q = Terrain(max_poly_degree, every_nth=steps[k])
            for j in range(n_degrees):
                """
                Loop over polynomial degrees.
                """
                r_score_train_tmp, mse_train_tmp, r_score_test_tmp, mse_test_tmp = \
                    q.standard_least_squares_regression(degree=j)

                r_score_train_avg[j] += r_score_train_tmp
                r_score_test_avg[j] += r_score_test_tmp
                mse_train_avg[j] += mse_train_tmp                
                mse_test_avg[j] += mse_test_tmp

        r_score_train_avg /= repetitions
        r_score_test_avg /= repetitions
        mse_train_avg /= repetitions
        mse_test_avg /= repetitions

        ax0[k].plot(degrees, mse_test_avg, color="black", label="test")
        ax0[k].plot(degrees, mse_train_avg, color="gray", linestyle="dashed", label="train")
        ax0[k].set_title(f"Data points: {q.n_data_points}")
        ax0[k].tick_params(labelsize=12)
        ax0[k].set_yticks(ticks=[5e4, 6e4, 7e4])
        ax0[k].set_ylim(4.5e4, 7.7e4)

        ax1[k].plot(degrees, r_score_train_avg, color="gray", linestyle="dashed", label="train")
        ax1[k].plot(degrees, r_score_test_avg, color="black", label="test")
        ax1[k].set_title(f"Data points: {q.n_data_points}")
        ax1[k].tick_params(labelsize=12)
        ax1[k].set_yticks(ticks=[-0.05, 0.05, 0.15, 0.25])
        ax1[k].set_ylim(-0.05-0.05, 0.25+0.05)

    ax0[1].set_yticklabels(labels=[])
    ax0[3].set_yticklabels(labels=[])
    ax0[0].set_xticklabels(labels=[])
    ax0[1].set_xticklabels(labels=[])
    ax0[1].legend(fontsize=12)

    ax1[1].set_yticklabels(labels=[])
    ax1[3].set_yticklabels(labels=[])
    ax1[0].set_xticklabels(labels=[])
    ax1[1].set_xticklabels(labels=[])
    ax1[1].legend(fontsize=12)

    plt.show()


if __name__ == "__main__":
    # ridge_cv()
    # lasso_cv()
    plain_ols()
