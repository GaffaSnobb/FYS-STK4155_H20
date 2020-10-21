import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import common

class _Solve:
    def gradient_descent(self, n_gradient_iterations, gradient_step_size):

        self.beta = np.zeros(self.poly_degree+1)
        for _ in range(n_gradient_iterations):
            gradient = self.X.T@(self.X@self.beta - self.y)*2/self.n_data_points
            self.beta -= gradient_step_size*gradient

    # def stochastic_gradient_descent(self, n_epochs):
    #     for epoch in range(n_epochs):
    #         for i in range(self.n_data_points):
    #             random_index = np.random.randint(self.n_data_points)
    #             xi = self.X[random_index:random_index+1]
    #             yi = self.y[random_index:random_index+1]
    #             gradients = 2 * xi.T @ ((xi @ theta)-yi)
    #             eta = learning_schedule(epoch*self.n_data_points+i)
    #             theta = theta - eta*gradients

class Example1D(_Solve):
    def __init__(self, n_data_points, poly_degree):
        self.n_data_points = n_data_points
        self.poly_degree = poly_degree
        self.n_features = common.features(self.poly_degree)
        
        self.x1 = np.random.uniform(0, 1, self.n_data_points)

        self.X = common.create_design_matrix_one_input_variable(self.x1,
            self.n_data_points, self.poly_degree)
        self.y = 2*self.x1 + 3*self.x1**2 + np.random.randn(self.n_data_points)

    def show(self):
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
        self.n_data_points = n_data_points
        self.poly_degree = poly_degree
        self.n_features = common.features(self.poly_degree)
        
        self.x1 = np.random.uniform(0, 1, self.n_data_points)
        self.x2 = np.random.uniform(0, 1, self.n_data_points)

        X = common.create_design_matrix(self.x1, self.x2, n_data_points, poly_degree)
        y = common.franke_function(self.x1, self.x2)
        beta = np.zeros(self.n_features)




if __name__ == "__main__":
    q = Example1D(n_data_points=100, poly_degree=3)
    q.gradient_descent(n_gradient_iterations=1000, gradient_step_size=0.3)
    q.show()
    pass