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
    def __init__(self, n_data_total, poly_degree, init_beta=None):
        """
        Set up a 2D example using Franke data.

        Parameters
        ----------
        n_data_total : int
            The number of data points.

        poly_degree : int
            The polynomial degree.
        """
        self.n_dependent_variables = 2
        self.x1 = np.random.uniform(0, 1, n_data_total)
        self.x2 = np.random.uniform(0, 1, n_data_total)

        self.X = common.create_design_matrix_two_dependent_variables(self.x1,
            self.x2, n_data_total, poly_degree)
        self.y = common.franke_function(self.x1, self.x2)
        
        super(Example2D, self).__init__(n_data_total, poly_degree, init_beta)


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


def visualise(beta, x, y, poly_degree, sgd_mse):
    """
    Compare a single fit with scipy.optimize.curve_fit.  For debugging.
    """
    n_scope = 1000
    scope = np.linspace(0, 1, n_scope)
    sgd_res = common.polynomial_1d(scope, *beta)

    popt, _ = curve_fit(f=common.polynomial_1d, xdata=x,
        ydata=y, p0=[0]*(poly_degree+1))

    curve_fit_mse = common.mean_squared_error(y, common.polynomial_1d(x, *popt))

    plt.plot(scope, sgd_res, label=f"SGD, MSE: {sgd_mse:.4f}")
    plt.plot(scope, common.polynomial_1d(scope, *popt),
        label=f"scipy.curve_fit, MSE: {curve_fit_mse:.4f}")
    plt.legend()
    plt.show()


def mse_vs_batches_no_ridge():
    n_data_total = 200
    n_epochs = 20
    n_repetitions = 10
    poly_degree = 3
    
    # q = Example1D(n_data_total, poly_degree)
    q = Example2D(n_data_total, poly_degree)

    batches = np.arange(1, n_data_total, 1)
    n_batches_total = len(batches)

    sgd_mse_train = np.zeros(n_batches_total)
    sgd_mse_test = np.zeros(n_batches_total)
    
    for rep in range(n_repetitions):
        """
        Repeat the experiment to get better data.
        """
        print(f"repetition {rep+1} of {n_repetitions}")
        for i in range(n_batches_total):
            """
            Loop over all step sizes.
            """
            q.stochastic_gradient_descent(n_epochs, n_batches=batches[i], lambd=0)
            sgd_mse_train_tmp, sgd_mse_test_tmp = q.mse
            sgd_mse_train[i] += sgd_mse_train_tmp
            sgd_mse_test[i] += sgd_mse_test_tmp

            if i == 0: beta = q.beta

    sgd_mse_train /= n_repetitions  # Average.
    sgd_mse_test /= n_repetitions

    plt.semilogy(batches, sgd_mse_train, label="train")
    plt.semilogy(batches, sgd_mse_test, label="test")
    plt.xlabel("batches")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    # visualise(beta, q.X_train[:, 1], q.y_train, poly_degree, sgd_mse_train[-1])


def mse_vs_step_size_no_ridge():
    n_data_total = 200
    n_epochs = 10
    n_repetitions = 20
    n_step_sizes = 100
    n_batches = 20
    poly_degree = 3
    step_sizes = np.linspace(1e-3, 1e-1, n_step_sizes)
    
    q = Example2D(n_data_total, poly_degree)
    sgd_mse_train = np.zeros(n_step_sizes)
    sgd_mse_test = np.zeros(n_step_sizes)
    
    for rep in range(n_repetitions):
        """
        Repeat the experiment to get better data.
        """
        print(f"repetition {rep+1} of {n_repetitions}")
        for i in range(n_step_sizes):
            """
            Loop over all batch sizes.
            """
            q.stochastic_gradient_descent(n_epochs, n_batches, step_sizes[i], lambd=0)
            sgd_mse_train_tmp, sgd_mse_test_tmp = q.mse
            sgd_mse_train[i] += sgd_mse_train_tmp
            sgd_mse_test[i] += sgd_mse_test_tmp

    sgd_mse_train /= n_repetitions  # Average.
    sgd_mse_test /= n_repetitions

    plt.semilogy(step_sizes, sgd_mse_train, label="train", color="black", linestyle="dashed")
    plt.semilogy(step_sizes, sgd_mse_test, label="test", color="black")
    plt.xlabel("Step size")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def mse_vs_step_size_vs_batches_no_ridge():
    n_data_total = 200
    n_epochs = 10
    n_repetitions = 2
    poly_degree = 3
    n_step_sizes = 30
    step_sizes = np.linspace(1e-3, 1e-1, n_step_sizes)
    batches = np.arange(1, n_data_total//2, 1)
    n_batches_total = len(batches)
    
    q = Example2D(n_data_total, poly_degree)
    sgd_mse_train = np.zeros((n_step_sizes, n_batches_total))
    sgd_mse_test = np.zeros((n_step_sizes, n_batches_total))
    
    for rep in range(n_repetitions):
        """
        Repeat the experiment to get better data.
        """
        # print(f"repetition {rep+1} of {n_repetitions}")
        for i in range(n_step_sizes):
            """
            Loop over all step sizes.
            """
            print(f"step size {i+1} of {n_step_sizes}: {step_sizes[i]}")
            print(f"repetition {rep+1} of {n_repetitions}\n")
            for j in range(n_batches_total):
                """
                Loop over all batch sizes.
                """

                q.stochastic_gradient_descent(n_epochs, batches[j],
                    step_sizes[i], lambd=0)
                sgd_mse_train_tmp, sgd_mse_test_tmp = q.mse
                sgd_mse_train[i, j] += sgd_mse_train_tmp
                sgd_mse_test[i, j] += sgd_mse_test_tmp

    sgd_mse_train /= n_repetitions  # Average.
    sgd_mse_test /= n_repetitions


    X, Y = np.meshgrid(batches, step_sizes)
    plt.contourf(X, Y, np.log10(sgd_mse_test))
    plt.xlabel("batches")
    plt.ylabel("step size")
    plt.colorbar()
    plt.show()


def mse_vs_lambda_vs_step_size_ridge():
    n_data_total = 200
    n_epochs = 10
    n_repetitions = 2
    n_batches = 20
    poly_degree = 3
    
    n_step_sizes = 200
    n_lambdas = 200
    step_sizes = np.linspace(1e-3, 1e-1, n_step_sizes)
    lambdas = np.linspace(0, 2, n_lambdas)
    
    q = Example2D(n_data_total, poly_degree)
    sgd_mse_train = np.zeros((n_step_sizes, n_lambdas))
    sgd_mse_test = np.zeros((n_step_sizes, n_lambdas))
    
    for rep in range(n_repetitions):
        """
        Repeat the experiment to get better data.
        """
        # print(f"repetition {rep+1} of {n_repetitions}")
        for i in range(n_step_sizes):
            """
            Loop over all step sizes.
            """
            print(f"step size {i+1} of {n_step_sizes}: {step_sizes[i]}")
            print(f"repetition {rep+1} of {n_repetitions}\n")
            for j in range(n_lambdas):
                """
                Loop over all rigde regression penalty parameters.
                """

                q.stochastic_gradient_descent(n_epochs, n_batches,
                    step_sizes[i], lambdas[j])
                sgd_mse_train_tmp, sgd_mse_test_tmp = q.mse
                sgd_mse_train[i, j] += sgd_mse_train_tmp
                sgd_mse_test[i, j] += sgd_mse_test_tmp

    sgd_mse_train /= n_repetitions  # Average.
    sgd_mse_test /= n_repetitions

    idx = np.unravel_index(np.argmin(sgd_mse_test), sgd_mse_test.shape)
    print(f"min. at lambda={lambdas[idx[0]]}, step={step_sizes[idx[1]]}")
    X, Y = np.meshgrid(lambdas, step_sizes)
    plt.contourf(X, Y, np.log10(sgd_mse_test))
    plt.plot(lambdas[idx[0]], step_sizes[idx[1]], "ro", label=f"min. at lambda={lambdas[idx[0]]:.4f}, step={step_sizes[idx[1]]}")
    plt.legend()
    plt.xlabel(r"$\lambda$")
    plt.ylabel("step size")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # mse_vs_batches_no_ridge()
    # mse_vs_step_size_no_ridge()
    # mse_vs_step_size_vs_batches_no_ridge()
    # mse_vs_lambda_vs_step_size_ridge()
    visualize_fit_1d()
    pass