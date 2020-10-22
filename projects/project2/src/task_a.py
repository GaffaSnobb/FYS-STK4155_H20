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
        
        for _ in range(iterations):
            """
            Loop over the gradient descents.
            """
            gradient = self.X.T@(self.X@self.beta - self.y)*2/self.n_data_points
            self.beta -= step_size*gradient

    def stochastic_gradient_descent(self, n_epochs, n_batches):
        """
        Solve for beta using stochastic gradient descent with momentum.

        Parameters
        ----------
        n_epochs : int
            The number of epochs.

        n_batches : int
            The number of batches.  If the number of rows in the design
            matrix does not divide by n_batches, the rest rows are
            discarded.
        """
        rest = self.n_data_points%n_batches # The rest after equally splitting X into batches.
        n_data_per_batch = self.n_data_points//n_batches # Index step size.
        # Indices of X corresponding to start point of the batches.
        batch_indices = np.arange(0, self.n_data_points-rest, n_data_per_batch)

        momentum_parameter = 0.5
        momentum = 0

        for epoch in range(n_epochs):
            """
            Loop over epochs.  For each loop, a random start index
            defined by the number of batches is drawn.  This chooses a
            random batch by slicing the design matrix.
            """
            random_index = np.random.choice(batch_indices)
            X = self.X[random_index:random_index+n_data_per_batch]
            y = self.y[random_index:random_index+n_data_per_batch]
            t = epoch*n_data_per_batch   # Does not need to be calculated in the inner loop.
            
            for i in range(n_data_per_batch):
                """
                Loop over all data in each batch.
                """
                t += i
                step_size = common.step_length(t=t, t0=5, t1=50)

                gradient = 2*X.T@((X@self.beta) - y)
                momentum = momentum_parameter*momentum + step_size*gradient
                self.beta -= momentum


class Example1D(_Solve):
    def __init__(self, n_data_points, poly_degree, init_beta=None):
        """
        Set up a 1D example for easy visualization of the process.

        Parameters
        ----------
        n_data_points : int
            The number of data points.

        poly_degree : int
            The polynomial degree.

        init_beta : NoneType, numpy.ndarray
            Initial beta values.  Defaults to None where 0 is used.
        """
        self.n_data_points = n_data_points
        self.poly_degree = poly_degree
        
        self.x1 = np.random.uniform(0, 1, self.n_data_points)
        self.X = common.create_design_matrix_one_input_variable(self.x1,
            self.n_data_points, self.poly_degree)
        self.y = 2*self.x1 + 3*self.x1**2 + np.random.randn(self.n_data_points)
        
        if init_beta is None:
            """
            Initial guess of beta.
            """
            self.beta = np.zeros(self.poly_degree+1)
        else:
            msg = "Initial beta value array must be of length"
            msg += f" {self.poly_degree + 1}, got {len(init_beta)}."
            success = len(init_beta) == (self.poly_degree+1)
            assert success, msg
            
            self.beta = init_beta


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

    q = Example1D(n_data_points=100, poly_degree=10)
    q.stochastic_gradient_descent(n_epochs=100, n_batches=10)
    # q.gradient_descent(iterations=1000, step_size=0.3)
    q.show()
    pass