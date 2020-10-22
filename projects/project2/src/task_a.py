import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import common


class Example1D(common._StatTools):
    def __init__(self, n_data_total, poly_degree, init_beta=None):
        """
        Set up a 1D example for easy visualization of the process.

        Parameters
        ----------
        n_data_total : int
            The total number of data points.

        poly_degree : int
            The polynomial degree.

        init_beta : NoneType, numpy.ndarray
            Initial beta values.  Defaults to None where 0 is used.
        """
        self.n_dependent_variables = 1
        
        self.x1 = np.random.uniform(0, 1, n_data_total)
        self.X = common.create_design_matrix_one_dependent_variable(self.x1,
            n_data_total, poly_degree)
        self.y = 2*self.x1 + 3*self.x1**2 + np.random.randn(n_data_total)        
        
        super(Example1D, self).__init__(n_data_total, poly_degree, init_beta)


class Example2D(common._StatTools):
    def __init__(self, n_data_total, poly_degree):
        """
        Set up a 2D example using Franke data.

        Parameters
        ----------
        n_data_total : int
            The number of data points.

        poly_degree : int
            The polynomial degree.
        """
        self.n_data_total = n_data_total
        self.poly_degree = poly_degree
        self.n_features = common.features(self.poly_degree)
        
        self.x1 = np.random.uniform(0, 1, self.n_data_total)
        self.x2 = np.random.uniform(0, 1, self.n_data_total)

        X = common.create_design_matrix(self.x1, self.x2, n_data_total,
            poly_degree)
        y = common.franke_function(self.x1, self.x2)
        beta = np.zeros(self.n_features)



def visualize_fit_1d():
    """
    Perform a fit on data of 1 dependent variable.  Plot SGD fit with
    scipy.optimize.curve_fit for comparison.
    """
    n_scope = 1000
    scope = np.linspace(0, 1, n_scope)
    
    q = Example1D(n_data_total=200, poly_degree=3)
    q.gradient_descent(iterations=1000, step_size=0.3)
    
    gd_res = common.polynomial_1d(scope, *q.beta)
    gd_mse = common.mean_squared_error(q.y_train, q.X_train@q.beta)
    q.reset_init_beta()
    
    q.stochastic_gradient_descent(n_epochs=100, n_batches=20, lambd=0)
    sgd_res = common.polynomial_1d(scope, *q.beta)
    sgd_mse = common.mean_squared_error(q.y_train, q.X_train@q.beta)

    # q.x1 = (q.x1 - q.X_mean)/q.X_std  # Scaling is not working yet.

    popt, _ = curve_fit(f=common.polynomial_1d, xdata=q.X_train[:, 1],
        ydata=q.y_train, p0=[0]*(q.poly_degree+1))

    curve_fit_mse = common.mean_squared_error(q.y_train,
        common.polynomial_1d(q.X_train[:, 1], *popt))


    plt.plot(q.x1, q.y, "r.")
    plt.plot(scope, gd_res, label=f"GD, MSE: {gd_mse:.4f}")
    plt.plot(scope, sgd_res, label=f"SGD, MSE: {sgd_mse:.4f}")
    plt.plot(scope, common.polynomial_1d(scope, *popt),
        label=f"scipy.curve_fit, MSE: {curve_fit_mse:.4f}")
    plt.legend()
    plt.show()


def compare_1d():
    n_data_total = 200
    n_epochs = 100
    n_repetitions = 10
    poly_degree = 3
    
    q = Example1D(n_data_total, poly_degree)

    batches = np.arange(1, n_data_total, 1)
    n_batches_total = len(batches)

    sgd_mse_train = np.empty(n_batches_total)
    sgd_mse_test = np.empty(n_batches_total)
    
    for i in range(n_batches_total):
        q.stochastic_gradient_descent(n_epochs, n_batches=batches[i], lambd=0)
        sgd_mse_train[i], sgd_mse_test[i] = q.mse

    plt.plot(batches, sgd_mse_test, label="test")
    plt.plot(batches, sgd_mse_train, label="train")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # compare_1d()
    visualize_fit_1d()
    pass