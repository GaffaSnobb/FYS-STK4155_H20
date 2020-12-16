import numpy as np
import matplotlib.pyplot as plt
import ray
import common

class Example2D(common._StatTools):
    def __init__(self, n_data_total, poly_degree, init_beta=None):
        """
        Set up a 2D example.

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


def mse_vs_epochs_batches_steps_lambdas():
    """
    Loop over epochs, step sizes and ridge regression parameters and
    calculate the MSE.  Use both our implementation and sklearn.  Plot
    the data.
    """
    n_data_total = 400
    n_repetitions = 30
    n_step_sizes = 10
    n_lambdas = 15
    poly_degree = 3
    lambdas = np.linspace(0, 0.1, n_lambdas)
    step_sizes = np.linspace(1e-3, 1e-1, n_step_sizes)
    epochs = np.arange(1, 20+1, 1)
    n_epochs = len(epochs)
    
    fname = f"data_files/task_a_mse_epochs_batches_steps_lambda_ridge"

    try:
        """
        Try to load the files.
        """
        sgd_mse_train = np.load(file=fname + "_mse_train.npy")
        sgd_mse_test = np.load(file=fname + "_mse_test.npy")
        sgd_mse_train_r = np.load(file=fname + "_r_train.npy")
        sgd_mse_test_r = np.load(file=fname + "_r_test.npy")

        sklearn_regression_mse_train = np.load(file=fname + "_sklearn_regression_mse_train.npy")
        sklearn_regression_mse_test = np.load(file=fname + "_sklearn_regression_mse_test.npy")
        sklearn_regression_ridge_mse_train = np.load(file=fname + "_sklearn_regression_ridge_mse_train.npy")
        sklearn_regression_ridge_mse_test = np.load(file=fname + "_sklearn_regression_ridge_mse_test.npy")

    except FileNotFoundError:
        """
        Generate and save the files if they dont exist.
        """
        regression = Example2D(n_data_total, poly_degree)
        regression_ridge = Example2D(n_data_total, poly_degree)
        
        sgd_mse_train = np.zeros((n_epochs, n_step_sizes))
        sgd_mse_test = np.zeros((n_epochs, n_step_sizes))
        sgd_mse_train_r = np.zeros((n_epochs, n_step_sizes, n_lambdas))
        sgd_mse_test_r = np.zeros((n_epochs, n_step_sizes, n_lambdas))
        sklearn_regression_mse_train = np.zeros((n_epochs, n_step_sizes))
        sklearn_regression_mse_test = np.zeros((n_epochs, n_step_sizes))
        sklearn_regression_ridge_mse_train = np.zeros((n_epochs, n_step_sizes, n_lambdas))
        sklearn_regression_ridge_mse_test = np.zeros((n_epochs, n_step_sizes,n_lambdas))

        ray.init()
        @ray.remote
        def epoch_steps_lambdas():
            """
            This function is parallelised by ray.
            """
            sgd_mse_train_tmp = np.zeros((n_epochs, n_step_sizes))
            sgd_mse_test_tmp = np.zeros((n_epochs, n_step_sizes))
            sgd_mse_train_r_tmp = np.zeros((n_epochs, n_step_sizes, n_lambdas))
            sgd_mse_test_r_tmp = np.zeros((n_epochs, n_step_sizes, n_lambdas))
            sklearn_regression_mse_train_tmp = np.zeros((n_epochs, n_step_sizes))
            sklearn_regression_mse_test_tmp = np.zeros((n_epochs, n_step_sizes))
            sklearn_regression_ridge_mse_train_tmp = np.zeros((n_epochs, n_step_sizes, n_lambdas))
            sklearn_regression_ridge_mse_test_tmp = np.zeros((n_epochs, n_step_sizes,n_lambdas))

            for e in range(n_epochs):
                """
                Loop over the number of epochs.
                """
                for s in range(n_step_sizes):
                    """
                    Loop over the number of step sizes.
                    """
                    regression.stochastic_gradient_descent(
                        n_epochs = epochs[e],
                        input_learning_rate = step_sizes[s],
                        lambd = 0)
                    
                    sgd_mse_train_tmp_tmp, sgd_mse_test_tmp_tmp = regression.mse
                    sgd_mse_train_tmp[e, s] += sgd_mse_train_tmp_tmp
                    sgd_mse_test_tmp[e, s] += sgd_mse_test_tmp_tmp

                    
                    regression.regression_with_sklearn(
                        n_epochs = epochs[e],
                        step = step_sizes[s])
                    
                    sklearn_regression_mse_train_tmp_tmp, sklearn_regression_mse_test_tmp_tmp = regression.mse_sklearn
                    sklearn_regression_mse_train_tmp[e, s] += sklearn_regression_mse_train_tmp_tmp
                    sklearn_regression_mse_test_tmp[e, s] += sklearn_regression_mse_test_tmp_tmp
                    
                    for l in range(n_lambdas):
                        """
                        Loop over the number of ridge regression
                        parameters.
                        """
                        print(f"\n repetition {rep + 1:3d} of {n_repetitions}")
                        print(f"{l + 1:3d} of {n_lambdas=}")
                        print(f"{s + 1:3d} of {n_step_sizes=}")
                        print(f"{e + 1:3d} of {n_epochs=}")
                        
                        regression_ridge.stochastic_gradient_descent(
                            n_epochs = epochs[e],
                            input_learning_rate = step_sizes[s],
                            lambd = lambdas[l])

                        sgd_mse_train_tmp_tmp_r, sgd_mse_test_tmp_tmp_r = regression_ridge.mse
                        sgd_mse_train_r_tmp[e, s, l] += sgd_mse_train_tmp_tmp_r
                        sgd_mse_test_r_tmp[e, s, l] += sgd_mse_test_tmp_tmp_r

                        regression_ridge.regression_with_sklearn(
                            n_epochs = epochs[e],
                            step = step_sizes[s],
                            alpha = lambdas[l])

                        sklearn_regression_ridge_mse_train_tmp_tmp, sklearn_regression_ridge_mse_test_tmp_tmp = regression_ridge.mse_sklearn
                        sklearn_regression_ridge_mse_train_tmp[e, s, l] += sklearn_regression_ridge_mse_train_tmp_tmp
                        sklearn_regression_ridge_mse_test_tmp[e, s, l] += sklearn_regression_ridge_mse_test_tmp_tmp


            return (sgd_mse_train_tmp, sgd_mse_test_tmp, sgd_mse_train_r_tmp,
                sgd_mse_test_r_tmp, sklearn_regression_mse_train_tmp, sklearn_regression_mse_test_tmp,
                sklearn_regression_ridge_mse_train_tmp, sklearn_regression_ridge_mse_test_tmp)

        parallel = []
        for rep in range(n_repetitions):
            """
            The different processes are created here.
            """
            parallel.append(epoch_steps_lambdas.remote())

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

    fig0, ax0 = plt.subplots(figsize=(9, 7))
    ax0.plot(epochs, sgd_mse_test[:, idx[1]], label="OLS", color="black")
    # ax0.plot(epochs, sgd_mse_test_r[:, idx_r[1], idx_r[2]], label="Ridge", color="grey")
    ax0.plot(epochs, sklearn_regression_mse_test[:, idx_skl[1]], label="scikit-learn OLS", color="grey", linestyle="solid")
    # ax0.plot(epochs,sklearn_regression_ridge_mse_test[:, idx_skl_r[1], idx_skl_r[2]], label="SKL_ridge", color="black",linestyle="dashed")
    ax0.set_xlabel("# epochs", fontsize=15)
    ax0.set_ylabel("MSE", fontsize=15)
    ax0.tick_params(labelsize=15)
    ax0.legend(fontsize=15)
    ax0.grid()
    fig0.savefig(dpi=300, fname="../fig/task_a_epochs_mse_sgd_OLS.png")
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.plot(step_sizes, sgd_mse_test[idx[0], :], label="OLS", color="black")
    # ax1.plot(step_sizes, sgd_mse_test_r[idx_r[0], :, idx_r[2]], label="Ridge", color="black")
    ax1.plot(step_sizes,sklearn_regression_mse_test[idx_skl[0], :], label="scikit-learn OLS", color="grey", linestyle="solid")
    # ax1.plot(step_sizes,sklearn_regression_ridge_mse_test[idx_skl_r[0], :, idx_skl_r[2]], label="SKL_ridge", color="black",linestyle="dashed")
    ax1.set_xlabel(r"$\eta$", fontsize=15)
    ax1.set_ylabel("MSE", fontsize=15)
    ax1.tick_params(labelsize=15)
    ax1.legend(fontsize=15)
    ax1.grid()
    fig1.savefig(dpi=300, fname="../fig/task_a_eta_mse_sgd_OLS.png")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    ax2.plot(lambdas, sgd_mse_test_r[idx_r[0], idx_r[1], :], label="Ridge", color="black")
    ax2.plot(lambdas,sklearn_regression_ridge_mse_test[idx_skl_r[0], idx_skl_r[1],:], label="scikit-learn ridge", color="grey")
    ax2.set_xlabel(r"$\lambda$", fontsize=15)
    ax2.set_ylabel("MSE", fontsize=15)
    ax2.tick_params(labelsize=15)
    ax2.legend(fontsize=15)
    ax2.grid()
    fig2.savefig(dpi=300, fname="../fig/task_a_lambda_mse_sgd_ridge")
    plt.show()
    print(f"min(MSE_OLS)={np.amin(sgd_mse_test):.5f} at epoch={epochs[idx[0]]}, learning rate={step_sizes[idx[1]]:.4f}")
    print(f"min(MSE_ridge)={np.amin(sgd_mse_test_r):.5f} at epoch={epochs[idx_r[0]]}, learning rate={step_sizes[idx_r[1]]:.4f}, lambda={lambdas[idx_r[2]]}")



if __name__ == "__main__":
    mse_vs_epochs_batches_steps_lambdas()
    pass