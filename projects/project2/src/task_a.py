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


    def show(self):
        """
        Plot and show the fit data and compare with
        scipy.optimize.curve_fit.
        """
        n_scope = 1000
        scope = np.linspace(0, 1, n_scope)
        res = common.polynomial_1d(scope, *self.beta)

        popt, pcov = curve_fit(f=common.polynomial_1d, xdata=self.x1,
            ydata=self.y, p0=[0]*(self.poly_degree+1))

        print(f"curve_fit: {popt}")
        print(f"gd: {self.beta}")

        plt.plot(self.x1, self.y, "r.")
        plt.plot(scope, res, label="gradient descent")
        plt.plot(scope, common.polynomial_1d(scope, *popt), label="curve_fit")
        plt.legend()
        plt.show()


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




if __name__ == "__main__":
    # q = Example1D(n_data_total=100, poly_degree=3)
    # q.gradient_descent(iterations=1000, step_size=0.3)
    # q.show()

    q = Example1D(n_data_total=200, poly_degree=3)
    q.stochastic_gradient_descent(n_epochs=100, n_batches=20, lambd=0)


    n_scope = 1000
    scope = np.linspace(0, 1, n_scope)
    res = common.polynomial_1d(scope, *q.beta)

    popt, pcov = curve_fit(f=common.polynomial_1d, xdata=q.x1,
        ydata=q.y, p0=[0]*(q.poly_degree+1))

    print(f"curve_fit: {popt}")
    print(f"gd: {q.beta}")

    plt.plot(q.x1, q.y, "r.")
    plt.plot(scope, res, label="gradient descent")
    plt.plot(scope, common.polynomial_1d(scope, *popt), label="curve_fit")
    plt.legend()
    plt.show()
    pass