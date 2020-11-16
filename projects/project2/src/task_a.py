import numpy as np
import matplotlib.pyplot as plt
import ray
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
    
    q.mini_batch_gradient_descent(n_epochs=100, n_batches=20, lambd=0)
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


def mse_vs_epochs_batches_steps_lambdas():
    """
    Marit.
    """
    # n_data_total = 400
    # n_repetitions = 30
    # n_step_sizes = 10
    # n_lambdas = 10
    # poly_degree = 3
    # lambdas = np.linspace(0, 0.1, n_lambdas)
    # step_sizes = np.linspace(1e-3, 1e-1, n_step_sizes)
    # batches = np.arange(1, 100+2, 5)
    # n_batches_total = len(batches)
    # epochs = np.arange(1, 20+1, 2)
    # n_epochs = len(epochs)
    n_data_total = 400
    n_repetitions = 15
    n_step_sizes = 10
    n_lambdas = 10
    poly_degree = 3
    lambdas = np.linspace(0, 0.1, n_lambdas)
    step_sizes = np.linspace(1e-3, 1e-1, n_step_sizes)
    batches = np.arange(1, 100+2, 10)
    n_batches_total = len(batches)
    epochs = np.arange(1, 20+1, 2)
    n_epochs = len(epochs)
    
    fname = f"task_a_mse_epochs_batches_steps_lambda_ridge"

    try:
        sgd_mse_train = np.load(file=fname + "_mse_train.npy")
        sgd_mse_test = np.load(file=fname + "_mse_test.npy")
        sgd_mse_train_r = np.load(file=fname + "_r_train.npy")
        sgd_mse_test_r = np.load(file=fname + "_r_test.npy")

        sklearn_regression_mse_train = np.load(file=fname + "_sklearn_regression_mse_train.npy")
        sklearn_regression_mse_test = np.load(file=fname + "_sklearn_regression_mse_test.npy")
        sklearn_regression_ridge_mse_train = np.load(file=fname + "_sklearn_regression_ridge_mse_train.npy")
        sklearn_regression_ridge_mse_test = np.load(file=fname + "_sklearn_regression_ridge_mse_test.npy")

    except FileNotFoundError:      
        # q = Example1D(n_data_total, poly_degree)
        regression = Example2D(n_data_total, poly_degree)
        regression_ridge = Example2D(n_data_total, poly_degree)
        
        sgd_mse_train = np.zeros((n_epochs, n_batches_total, n_step_sizes))
        sgd_mse_test = np.zeros((n_epochs, n_batches_total, n_step_sizes))
        sgd_mse_train_r = np.zeros((n_epochs, n_batches_total, n_step_sizes, n_lambdas))
        sgd_mse_test_r = np.zeros((n_epochs, n_batches_total, n_step_sizes, n_lambdas))
        sklearn_regression_mse_train = np.zeros((n_epochs, n_batches_total, n_step_sizes))
        sklearn_regression_mse_test = np.zeros((n_epochs, n_batches_total, n_step_sizes))
        sklearn_regression_ridge_mse_train = np.zeros((n_epochs, n_batches_total, n_step_sizes, n_lambdas))
        sklearn_regression_ridge_mse_test = np.zeros((n_epochs, n_batches_total, n_step_sizes,n_lambdas))

        ray.init()
        @ray.remote
        def func():
            sgd_mse_train_tmp = np.zeros((n_epochs, n_batches_total, n_step_sizes))
            sgd_mse_test_tmp = np.zeros((n_epochs, n_batches_total, n_step_sizes))
            sgd_mse_train_r_tmp = np.zeros((n_epochs, n_batches_total, n_step_sizes, n_lambdas))
            sgd_mse_test_r_tmp = np.zeros((n_epochs, n_batches_total, n_step_sizes, n_lambdas))
            sklearn_regression_mse_train_tmp = np.zeros((n_epochs, n_batches_total, n_step_sizes))
            sklearn_regression_mse_test_tmp = np.zeros((n_epochs, n_batches_total, n_step_sizes))
            sklearn_regression_ridge_mse_train_tmp = np.zeros((n_epochs, n_batches_total, n_step_sizes, n_lambdas))
            sklearn_regression_ridge_mse_test_tmp = np.zeros((n_epochs, n_batches_total, n_step_sizes,n_lambdas))

            for e in range(n_epochs):
                """
                Loop over the number of epochs.
                """
                for b in range(n_batches_total):
                    """
                    Loop over the number of batches.
                    """
                    for s in range(n_step_sizes):
                        """
                        Loop over the number of step sizes.
                        """
                        # regression.mini_batch_gradient_descent(int(epochs[e]), n_batches=batches[b], input_step_size=step_sizes[s], lambd=0)
                        regression.stochastic_gradient_descent(
                            n_epochs = epochs[e],
                            input_learning_rate = step_sizes[s],
                            lambd = 0)
                        
                        sgd_mse_train_tmp_tmp, sgd_mse_test_tmp_tmp = regression.mse
                        sgd_mse_train_tmp[e, b, s] += sgd_mse_train_tmp_tmp
                        sgd_mse_test_tmp[e, b, s] += sgd_mse_test_tmp_tmp

                        
                        regression.regression_with_sklearn(
                            n_epochs = epochs[e],
                            step = step_sizes[s])
                        
                        sklearn_regression_mse_train_tmp_tmp, sklearn_regression_mse_test_tmp_tmp = regression.mse_sklearn
                        sklearn_regression_mse_train_tmp[e, b, s] += sklearn_regression_mse_train_tmp_tmp
                        sklearn_regression_mse_test_tmp[e, b, s] += sklearn_regression_mse_test_tmp_tmp
                        
                        for l in range(n_lambdas):
                            """
                            Loop over the number of ridge regression
                            parameters.
                            """
                            print(f"\n repetition {rep + 1:3d} of {n_repetitions}")
                            print(f"{l + 1:3d} of {n_lambdas=}")
                            print(f"{s + 1:3d} of {n_step_sizes=}")
                            print(f"{b + 1:3d} of {n_batches_total=}")
                            print(f"{e + 1:3d} of {n_epochs=}")
                            
                            # regression_ridge.mini_batch_gradient_descent(
                            #     n_epochs = int(epochs[e]),
                            #     n_batches = batches[b],
                            #     input_step_size = step_sizes[s],
                            #     lambd = lambdas[l])
                            regression_ridge.stochastic_gradient_descent(
                                n_epochs = epochs[e],
                                input_learning_rate = step_sizes[s],
                                lambd = lambdas[l])

                            sgd_mse_train_tmp_tmp_r, sgd_mse_test_tmp_tmp_r = regression_ridge.mse
                            sgd_mse_train_r_tmp[e, b, s, l] += sgd_mse_train_tmp_tmp_r
                            sgd_mse_test_r_tmp[e, b, s, l] += sgd_mse_test_tmp_tmp_r

                            regression_ridge.regression_with_sklearn(
                                n_epochs = epochs[e],
                                step = step_sizes[s],
                                alpha = lambdas[l])

                            sklearn_regression_ridge_mse_train_tmp_tmp, sklearn_regression_ridge_mse_test_tmp_tmp = regression_ridge.mse_sklearn
                            sklearn_regression_ridge_mse_train_tmp[e, b, s, l] += sklearn_regression_ridge_mse_train_tmp_tmp
                            sklearn_regression_ridge_mse_test_tmp[e, b, s, l] += sklearn_regression_ridge_mse_test_tmp_tmp

                    break
                    # if b == 0: beta = regression.beta

            return (sgd_mse_train_tmp, sgd_mse_test_tmp, sgd_mse_train_r_tmp,
                sgd_mse_test_r_tmp, sklearn_regression_mse_train_tmp, sklearn_regression_mse_test_tmp,
                sklearn_regression_ridge_mse_train_tmp, sklearn_regression_ridge_mse_test_tmp)

        parallel = []
        for rep in range(n_repetitions):
            """
            The different processes are created here.
            """
            parallel.append(func.remote())

        for res in ray.get(parallel):
            """
            The parallel work is performed and extracted here.
            """
            (sgd_mse_train_tmp, sgd_mse_test_tmp, sgd_mse_train_r_tmp,
                sgd_mse_test_r_tmp, sklearn_regression_mse_train_tmp, sklearn_regression_mse_test_tmp,
                sklearn_regression_ridge_mse_train_tmp, sklearn_regression_ridge_mse_test_tmp) = res
            
            sgd_mse_train += sgd_mse_train_tmp
            sgd_mse_test += sgd_mse_test_tmp
            sgd_mse_train_r += sgd_mse_train_r_tmp
            sgd_mse_test_r += sgd_mse_test_r_tmp

            sklearn_regression_mse_train += sklearn_regression_mse_train_tmp
            sklearn_regression_mse_test += sklearn_regression_mse_test_tmp
            sklearn_regression_ridge_mse_train += sklearn_regression_ridge_mse_train_tmp
            sklearn_regression_ridge_mse_test += sklearn_regression_ridge_mse_test_tmp


        sgd_mse_train /= n_repetitions  # Average.
        sgd_mse_test /= n_repetitions
        sgd_mse_train_r /= n_repetitions
        sgd_mse_test_r /= n_repetitions
        sklearn_regression_mse_train /= n_repetitions
        sklearn_regression_mse_test /= n_repetitions
        sklearn_regression_ridge_mse_train /= n_repetitions
        sklearn_regression_ridge_mse_test /= n_repetitions

        np.save(file=fname + "_mse_train.npy", arr=sgd_mse_train)
        np.save(file=fname + "_mse_test.npy", arr=sgd_mse_test)
        np.save(file=fname + "_r_train.npy", arr=sgd_mse_train_r)
        np.save(file=fname + "_r_test.npy", arr=sgd_mse_test_r)

        np.save(file=fname + "_sklearn_regression_mse_train.npy", arr=sklearn_regression_mse_train)
        np.save(file=fname + "_sklearn_regression_mse_test.npy", arr=sklearn_regression_mse_test)
        np.save(file=fname + "_sklearn_regression_ridge_mse_train.npy", arr=sklearn_regression_ridge_mse_train)
        np.save(file=fname + "_sklearn_regression_ridge_mse_test.npy", arr=sklearn_regression_ridge_mse_test)

    

    idx = np.unravel_index(np.argmin(sgd_mse_test), sgd_mse_test.shape)
    idx_r = np.unravel_index(np.argmin(sgd_mse_test_r), sgd_mse_test_r.shape)
    idx_skl = np.unravel_index(np.argmin(sklearn_regression_mse_test), sklearn_regression_mse_test.shape)
    idx_skl_r = np.unravel_index(np.argmin(sklearn_regression_ridge_mse_test), sklearn_regression_ridge_mse_test.shape)

    plt.plot(epochs, sgd_mse_test[:, idx[1], idx[2]], label="OLS", color="grey")
    plt.plot(epochs, sgd_mse_test_r[:, idx_r[1], idx_r[2], idx_r[3]], label="Ridge", color="black")
    plt.plot(epochs, sklearn_regression_mse_test[:,idx_skl[1],idx_skl[2]], label="SKL", color="grey", linestyle="dashed")
    plt.plot(epochs,sklearn_regression_ridge_mse_test[:,idx_skl_r[1],idx_skl_r[2],idx_skl_r[3]], label="SKL_ridge", color="black",linestyle="dashed")
    plt.xlabel("Number of epochs", fontsize=15)
    plt.ylabel("MSE", fontsize=15)
    plt.title("Stochastic gradient descent", fontsize=15)
    plt.tick_params(labelsize=12)
    plt.legend()
    #plt.savefig(dpi=300, fname="task_a_epochs_mse_OLS_ridge")
    plt.show()

    # plt.plot(batches, sgd_mse_test[idx[0],:,idx[2]], label="OLS", color="grey")
    # plt.plot(batches, sgd_mse_test_r[idx_r[0],:,idx[2],idx_r[3]], label="Ridge", color="black")
    # plt.xlabel("Number of mini-batches", fontsize=15)
    # plt.ylabel("MSE", fontsize=15)
    # plt.title("Stochastic gradient descent", fontsize=15)
    # plt.tick_params(labelsize=12)
    # plt.legend()
    # #plt.savefig(dpi=300, fname="task_a_batches_mse_sgd_OLS_ridge")
    # plt.show()

    plt.plot(step_sizes, sgd_mse_test[idx[0],idx[1],:], label="OLS", color="grey")
    plt.plot(step_sizes, sgd_mse_test_r[idx_r[0],idx_r[1],:,idx_r[3]], label="Ridge", color="black")
    plt.plot(step_sizes,sklearn_regression_mse_test[idx_skl[0],idx_skl[1],:], label="SKL", color="grey",linestyle="dashed")
    plt.plot(step_sizes,sklearn_regression_ridge_mse_test[idx_skl_r[0],idx_skl_r[1],:,idx_skl_r[3]], label="SKL_ridge", color="black",linestyle="dashed")
    plt.xlabel("Learning rates", fontsize=15)
    plt.ylabel("MSE", fontsize=15)
    plt.title("Stochastic gradient descent", fontsize=15)
    plt.tick_params(labelsize=12)
    plt.legend()
    #plt.savefig(dpi=300, fname="task_a_stepsize_mse_sgd_OLS_ridge")
    plt.show()

    plt.plot(lambdas, sgd_mse_test_r[idx_r[0],idx_r[1],idx_r[2],:], label="Ridge", color="black")
    plt.plot(lambdas,sklearn_regression_ridge_mse_test[idx_skl_r[0],idx_skl_r[1],idx_skl_r[2],:], label="SKL_ridge", color="grey")
    plt.xlabel("Lambdas", fontsize=15)
    plt.ylabel("MSE", fontsize=15)
    plt.title("Stochastic gradient descent, ridge", fontsize=15)
    plt.tick_params(labelsize=12)
    #plt.savefig(dpi=300, fname="task_a_lambda_mse_sgd_ridge")
    plt.show()
    print(f"min(MSE_OLS)={np.amin(sgd_mse_test)} at epoch={epochs[idx[0]]}, batch#={batches[idx[1]]}, learning rate={step_sizes[idx[2]]}")
    print(f"min(MSE_ridge)={np.amin(sgd_mse_test_r)} at epoch={epochs[idx_r[0]]},  batch#={batches[idx_r[1]]}, learning rate={step_sizes[idx_r[2]]}, lambda={lambdas[idx_r[3]]}")


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
            q.mini_batch_gradient_descent(n_epochs, n_batches=batches[i], lambd=0)
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
            q.mini_batch_gradient_descent(n_epochs, n_batches, step_sizes[i], lambd=0)
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

                q.mini_batch_gradient_descent(n_epochs, batches[j],
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


def mse_vs_lambda_ridge():
    n_data_total = 400
    n_epochs = 10
    n_repetitions = 1
    n_batches = 20
    poly_degree = 3
    
    n_lambdas = 100
    lambdas = np.linspace(0, 2, n_lambdas)
    
    q = Example2D(n_data_total, poly_degree)
    sgd_mse_train = np.zeros(n_lambdas)
    sgd_mse_test = np.zeros(n_lambdas)
    
    for rep in range(n_repetitions):
        """
        Repeat the experiment to get better data.
        """
        # print(f"repetition {rep+1} of {n_repetitions}")
        for j in range(n_lambdas):
            """
            Loop over all rigde regression penalty parameters.
            """

            q.mini_batch_gradient_descent(n_epochs, n_batches,
                0.07, lambdas[j])
            sgd_mse_train_tmp, sgd_mse_test_tmp = q.mse
            sgd_mse_train[j] += sgd_mse_train_tmp
            sgd_mse_test[j] += sgd_mse_test_tmp

    sgd_mse_train /= n_repetitions  # Average.
    sgd_mse_test /= n_repetitions

    idx = np.argmin(sgd_mse_test)
    print(f"min. at lambda={lambdas[idx]}, mse={sgd_mse_test[idx]}")

    plt.plot(lambdas, sgd_mse_test)
    plt.legend()
    plt.xlabel(r"$\lambda$")
    plt.ylabel("step size")
    plt.show()


def mse_vs_lambda_vs_step_size_ridge():
    n_data_total = 200
    n_epochs = 10
    n_repetitions = 15
    n_batches = 20
    poly_degree = 3
    
    n_step_sizes = 200
    n_lambdas = 200
    step_sizes = np.linspace(1e-5, 1e-1, n_step_sizes)
    lambdas = np.linspace(0, 2, n_lambdas)
    fname = f"task_a_mse_lambda_step_ridge"

    try:
        sgd_mse_train = np.load(file=fname + "train.npy")
        sgd_mse_test = np.load(file=fname + "test.npy")
    
    except FileNotFoundError:
        q = Example2D(n_data_total, poly_degree)
        sgd_mse_train = np.zeros((n_step_sizes, n_lambdas))
        sgd_mse_test = np.zeros((n_step_sizes, n_lambdas))

        ray.init()
        @ray.remote
        def step_and_lambda_loop():
            sgd_mse_train_tmp = np.zeros((n_step_sizes, n_lambdas))
            sgd_mse_test_tmp = np.zeros((n_step_sizes, n_lambdas))
            for i in range(n_step_sizes):
                """
                Loop over all step sizes.
                """
                for j in range(n_lambdas):
                    """
                    Loop over all rigde regression penalty parameters.
                    """
                    print(f"\nrepetition {rep + 1} of {n_repetitions}\n")
                    print(f"step size {i + 1} of {n_step_sizes}: {step_sizes[i]=}")
                    print(f"lambda {j + 1} of {n_lambdas}: {lambdas[j]=}")

                    q.mini_batch_gradient_descent(n_epochs, n_batches,
                        step_sizes[i], lambdas[j])
                    sgd_mse_train_tmp_tmp, sgd_mse_test_tmp_tmp = q.mse
                    sgd_mse_train_tmp[i, j] += sgd_mse_train_tmp_tmp
                    sgd_mse_test_tmp[i, j] += sgd_mse_test_tmp_tmp

            return sgd_mse_train_tmp, sgd_mse_test_tmp
        
        parallel = []
        for rep in range(n_repetitions):
            """
            The different processes are created here.
            """
            parallel.append(step_and_lambda_loop.remote())

        for res in ray.get(parallel):
            """
            The parallel work is performed and extracted here.
            """
            sgd_mse_train_tmp, sgd_mse_test_tmp = res
            sgd_mse_train += sgd_mse_train_tmp
            sgd_mse_test += sgd_mse_test_tmp


        sgd_mse_train /= n_repetitions  # Average.
        sgd_mse_test /= n_repetitions
        
        np.save(file=fname + "train.npy", arr=sgd_mse_train)
        np.save(file=fname + "test.npy", arr=sgd_mse_test)

    idx = np.unravel_index(np.argmin(sgd_mse_test), sgd_mse_test.shape)
    print(f"min. at lambda={lambdas[idx[1]]}, step={step_sizes[idx[0]]}, {sgd_mse_test[idx]=}")
    
    X, Y = np.meshgrid(lambdas, step_sizes)
    plt.contourf(X, Y, (sgd_mse_test))
    plt.plot(lambdas[idx[1]], step_sizes[idx[0]], "ro", label=f"min. at lambda={lambdas[idx[1]]:.4f}, step={step_sizes[idx[0]]}")
    plt.legend()
    plt.xlabel(r"$\lambda$")
    plt.ylabel("step size")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # mse_vs_lambda_ridge()
    # mse_vs_lambda_vs_step_size_ridge()
    mse_vs_epochs_batches_steps_lambdas()
    pass