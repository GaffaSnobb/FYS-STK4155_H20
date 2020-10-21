import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import common

class _Solve:
    def gradient_descent(self, iterations, step_size):
        """
        Solve for beta using gradient descent.

        Parameters
        ----------
        iterations : int
            The number of iterations of the gradient descent.
        
        step_size : int
            The step size of the gradient descent.  AKA learning rate.
        """
        self.beta = np.zeros(self.poly_degree+1)    # Initial 'guess'.
        for _ in range(iterations):
            """
            Loop over the gradient descents.
            """
            gradient = self.X.T@(self.X@self.beta - self.y)*2/self.n_data_points
            self.beta -= step_size*gradient

    def stochastic_gradient_descent(self, n_epochs, n_batches):
        n_rows = self.X.shape[0]
        print(n_rows)
        print(n_rows%n_batches)

        rest = n_rows%n_batches
        end_idx = n_rows - rest

        batch_indices = np.arange(0, n_rows-rest, n_rows//n_batches)
        print(f"n_data_points: {self.n_data_points}")
        print(f"n_batches: {n_batches}")
        print(batch_indices)

        # batches = np.split(self.X[:end_idx], n_batches)

        # if rest != 0:
        #     """
        #     Include the rest rows in the final batch.
        #     """
        #     tmp_rows = n_rows//n_batches + n_rows%n_batches
        #     tmp_cols = self.poly_degree + 1
        #     tmp = np.zeros(shape=(tmp_rows, tmp_cols))
        #     tmp[:n_rows//n_batches] = batches[-1]
        #     tmp[n_rows//n_batches:] = self.X[end_idx:]


        # for epoch in range(n_epochs):
        #     """
        #     Loop over all epochs.
        #     """
        #     for i in range(self.n_data_points):
        #         random_index = np.random.randint(self.n_data_points)
        #         X = self.X[random_index:random_index+1]
        #         y = self.y[random_index:random_index+1]
        #         gradients = 2*X.T@((X@beta) - y)
        #         eta = learning_schedule(epoch*self.n_data_points+i)
        #         beta = beta - eta*gradients


class Example1D(_Solve):
    def __init__(self, n_data_points, poly_degree):
        """
        Set up a 1D example for easy visualization of the process.

        Parameters
        ----------
        n_data_points : int
            The number of data points.

        poly_degree : int
            The polynomial degree.
        """
        self.n_data_points = n_data_points
        self.poly_degree = poly_degree
        self.n_features = common.features(self.poly_degree)
        
        self.x1 = np.random.uniform(0, 1, self.n_data_points)

        self.X = common.create_design_matrix_one_input_variable(self.x1,
            self.n_data_points, self.poly_degree)
        self.y = 2*self.x1 + 3*self.x1**2 + np.random.randn(self.n_data_points)

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


class Example2D(_Solve):
    def __init__(self, n_data_points, poly_degree):
        """
        Set up a 2D example using Franke data.

        Parameters
        ----------
        n_data_points : int
            The number of data points.

        poly_degree : int
            The polynomial degree.
        """
        self.n_data_points = n_data_points
        self.poly_degree = poly_degree
        self.n_features = common.features(self.poly_degree)
        
        self.x1 = np.random.uniform(0, 1, self.n_data_points)
        self.x2 = np.random.uniform(0, 1, self.n_data_points)

        X = common.create_design_matrix(self.x1, self.x2, n_data_points,
            poly_degree)
        y = common.franke_function(self.x1, self.x2)
        beta = np.zeros(self.n_features)




if __name__ == "__main__":
    # q = Example1D(n_data_points=100, poly_degree=3)
    # q.gradient_descent(iterations=1000, step_size=0.3)
    # q.show()

    q = Example1D(n_data_points=10, poly_degree=3)
    q.stochastic_gradient_descent(n_epochs=10, n_batches=2)
    pass