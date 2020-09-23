import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def franke_function(x1, x2):
    return 0.75*np.exp(-(0.25*(9*x1 - 2)**2) - 0.25*((9*x2 - 2)**2)) \
        + 0.75*np.exp(-((9*x1 + 1)**2)/49.0 - 0.1*(9*x2 + 1)) \
        + 0.5*np.exp(-(9*x1 - 7)**2/4.0 - 0.25*((9*x2 - 3)**2)) \
        - 0.2*np.exp(-(9*x1 - 4)**2 - (9*x2 - 7)**2)


def mean_squared_error(y_observed, y_predicted):
    """
    Calculate the mean squared error.

    Consider adding the length n as an argument if this function is
    called many times.

    Parameters
    ----------
    y_observed : numpy.ndarray
        Observed values.

    y_predicted : numpy.ndarray, float
        predicted values.

    Returns
    -------
    : numpy.ndarray
        The mean squared error.
    """
    # return np.sum((y_observed - y_predicted)**2)/len(y_observed)
    return np.mean((y_observed - y_predicted)**2)


def bias(f, y):
    """
    Calculate the bias.

    Parameters
    ----------
    f : numpy.ndarray
        Function values.

    y : numpy.ndarray

    Returns
    -------
    : numpy.ndarray
        The bias.
    """
    return mean_squared_error(f, np.mean(y))


def r_squared(y_observed, y_predicted):
    """
    Calculate the score R**2.

    Parameters
    ----------
    y_observed : numpy.ndarray
        Observed values.

    y_predicted : numpy.ndarray
        predicted values.

    Returns
    -------
    : numpy.ndarray
        The R**2 score.
    """
    return 1 - np.sum((y_observed - y_predicted)**2)/\
        np.sum((y_observed - np.mean(y_observed))**2)


class Solve:
    def __init__(self, deg, N=34, noise_factor=0.15, draw_random=False,
        debug_info=False, timing_info=False):
        """
        Solve the OLS on the Franke function.

        Draw N numbers in the inverval [0, 1) for both variables, x1 and x2.
        Make a meshgrid of x1 and x2 to ensure all combinations of x1 and x2
        values.  Pass the meshgrids to the Franke function, ravel the
        resulting array for easier calculations and add stochastic noise
        drawn from the standard normal distribution. Create the design
        matrix X based on the meshgrids, the number of data points and
        the polynomial degree.

        Parameters
        ----------
        deg : int
            Polynomial degree.

        N : int
            The number of data points per random variable.  The resulting
            meshgrids will measure NxN values.

        noise_factor : int, float
            The factor of added stochastic noise.

        draw_random : boolean
            If True, x1 and x2 values will be drawn from the standard
            normal distribution.  If False, linspace is used.

        debug_info : boolean
            For toggling print of debug data on / off.

        timing_info : boolean
            For toggling print of timing info on / off.
        """
        self.debug_info = debug_info
        self.timing_info = timing_info
        self.split_scale_called = False

        if draw_random:
            x1, x2 = np.meshgrid(np.random.randn(N), np.random.randn(N))
        else:
            x1, x2 = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

        self.y_observed = franke_function(x1, x2).ravel()
        self.y_observed += noise_factor*np.random.randn(N**2) # Stochastic noise.
        
        create_time = time.time()
        self.X = self._create_design_matrix(x1, x2, N, deg)
        create_time = time.time() - create_time

        if self.timing_info:
            print(f"design matrix created in {create_time:.3f} s")
            print(f"design matrix dimensions {self.X.shape}")

        self._split_scale()


    def _split_scale(self):
        """
        Split the data into training and test sets.  Scale the data by
        subtracting the mean and dividing by the standard deviation,
        both values from the training set.
        """
        # Splitting.
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y_observed, test_size=0.2)

        # Scaling.
        X_mean = np.mean(self.X_train)
        X_std = np.std(self.X_train)
        self.X_train = (self.X_train - X_mean)/X_std
        self.X_test = (self.X_test - X_mean)/X_std

        self.split_scale_called = True


    def _create_design_matrix(self, x1, x2, N, deg):
        """
        Construct a design matrix with N**2 rows and features =
        (deg + 1)*(deg + 2)/2 columns.  N**2 is the number of samples and
        features is the number of features of the design matrix.

        Parameters
        ----------

        x1 : numpy.ndarray
            Dependent / outcome / response variable. Is it, though?

        x2 : numpy.ndarray
            Dependent / outcome / response variable. Is it, though?

        N : int
            The number of randomly drawn data ponts per variable.

        deg : int
            The polynomial degree.

        Returns
        -------
        X : numpy.ndarray
            Design matrix of dimensions N**2 rows and (deg + 1)*(deg + 2)/2
            columns.
        """

        x1 = x1.ravel()
        x2 = x2.ravel()
        
        self.features = int((deg + 1)*(deg + 2)/2)
        X = np.empty((N**2, self.features))     # Data points x features.
        X[:, 0] = 1 # Intercept.
        col_idx = 1 # For indexing the design matrix columns.

        for j in range(1, deg+1):
            """
            Loop over all degrees except 0.
            """
            for k in range(j+1):
                """
                Loop over all combinations of x1 and x2 which produces
                an j'th degree term.
                """
                X[:, col_idx] = (x1**(j - k))*x2**k
                col_idx += 1

        return X


    def bootstrap(self, n_bootstraps=50):
        """
        Perform the OLS with bootstrapping.

        Parameters
        ----------
        n_bootstraps : int
            The number of bootstrap samples.

        Returns
        -------
        r_score_train : float
            The R^2 value of the training set.
        
        mse_train : float
            The mean squared error of the training set.
        """
        Y_predicted = np.empty((self.X_test.shape[0], n_bootstraps))
        beta = np.empty((self.X_test.shape[1], n_bootstraps))   # May not be necessary to store all betas.

        for b in range(n_bootstraps):
            """
            Draw 'n_bootstrap' bootstrap resamples and calculate
            predicted y values based on every resample.
            """
            X_train_resample, y_train_resample = resample(self.X_train,
                self.y_train, replace=True)

            inversion_time = time.time()
            beta[:, b] = np.linalg.pinv(X_train_resample.T@X_train_resample)@\
                X_train_resample.T@y_train_resample
            # beta[:, b] = ols(X_train_resample, y_train_resample, lambd=1)
            inversion_time = time.time() - inversion_time

            if self.timing_info:
                print(f"solved for beta in {inversion_time:.3f} s")

            Y_predicted[:, b] = self.X_test@beta[:, b]
        
        y_predicted = np.mean(Y_predicted, axis=1)  # Average over all columns.
        
        self.r_score_test_boot = r_squared(self.y_test, y_predicted)
        self.bias_test_boot = np.mean( (self.y_test - np.mean(Y_predicted, axis=1))**2 )
        self.variance_boot = np.mean(np.var(Y_predicted, axis=1))
        self.mse_test_boot = np.mean((self.y_test.reshape(-1, 1) - Y_predicted)**2)


        if self.debug_info:
            print(self.r_score_test_boot)
            print(self.bias_test_boot)
            print(self.variance_boot)
            print(self.mse_test_boot)


    def no_bootstrap(self):
        """
        Perform the OLS with no bootstrapping.

        Solve for the vector beta by matrix inversion and matrix
        multiplication.  Use the beta vector to generate model data
        (y_tilde) and predicted data (y_predicted). Return the R^2 score
        and MSE for both training and test data sets.

        Returns
        -------
        r_score_train : float
            The R^2 value of the training set.
        
        mse_train : float
            The mean squared error of the training set.

        r_score_test : float
            The R^2 value of the test set.

        mse_test : float
            The mean squared error of the test set.
        """
        inversion_time = time.time()
        beta = np.linalg.pinv(self.X_train.T@self.X_train)@self.X_train.T@self.y_train
        inversion_time = time.time() - inversion_time
        if self.timing_info:
            print(f"solved for beta in {inversion_time:.3f} s")

        y_model = self.X_train@beta
        y_predicted = self.X_test@beta

        r_score_train = r_squared(self.y_train, y_model)
        r_score_test = r_squared(self.y_test, y_predicted)
        mse_train = mean_squared_error(self.y_train, y_model)
        mse_test = mean_squared_error(self.y_test, y_predicted)

        if self.debug_info:
            print("\ntrain")
            print(f"R^2: {r_score_train}")
            print(f"MSE: {mse_train}")
            
            print("train (sklearn)")
            print(f"R^2: {skl.r2_score(y_train, y_model)}")
            print(f"MSE: {skl.mean_squared_error(y_train, y_model)}")
            
            print("\ntest")
            print(f"R^2: {r_score_test}")
            print(f"MSE: {mse_test}")
            
            print("test (sklearn)")
            print(f"R^2: {skl.r2_score(y_test, y_predicted)}")
            print(f"MSE: {skl.mean_squared_error(y_test, y_predicted)}")

        return r_score_train, mse_train, r_score_test, mse_test


    def cross_validation(self, folds, lots_of_info=False):
        """
        Perform the OLS with k-fold cross validation.

        Parameters
        ----------
        folds : int
            The number of folds.

        lots_of_info : boolean
            Toggle print of a bunch of debug info on / off.  Remove
            before final delivery.
        """
        sample_length = self.X_train.shape[0]//folds
        rest = self.X_train.shape[0]%folds
        total_sample_data_points = self.X_train.shape[0] - rest  # Removes 'rest' amount of trailing data points.
        
        # np.random.shuffle(self.X)   # In-place shuffle the rows.
        X_train_sample = np.empty((total_sample_data_points-sample_length, self.features))
        X_validation_sample = np.empty((sample_length, self.features))  # Not actually in use for task 1c.
        
        y_train_sample = np.empty(total_sample_data_points-sample_length)
        Y_validation_sample = np.empty((sample_length, folds))  # Not actually in use for task 1c.
        Y_predicted = np.empty((self.X_test.shape[0], folds))  # Predictions without the 'holy' test set.
        
        self.beta_cv = np.empty((self.features, folds))

        if lots_of_info:
            print(f"rest: {rest}")
            print(f"\nsample length: {sample_length}")
            print(f"true sample length: {self.X_train.shape[0]/folds}")
            print(f"\nX_train_sample dim: {X_train_sample.shape}")
            print(f"X_validation_sample dim: {X_validation_sample.shape}")
        
        for i in range(folds):
            """
            Split the training data into equally sized portions.
            """
            validation_start = i*sample_length
            validation_stop = (i + 1)*sample_length
            X_validation_sample[:, :] = \
                self.X_train[validation_start:validation_stop]
            Y_validation_sample[:, i] = \
                self.y_train[validation_start:validation_stop]

            if lots_of_info:
                print(f"\ni = {i}")
                print(f"validation: [{validation_start}:{validation_stop}]")
            
            if i > 0:
                """
                Validation subsample is not located at the beginning of
                X.
                """
                if lots_of_info:
                    print("i > 0")
                    print(f"\t[0:{validation_start}] len = {validation_start}")
                
                y_train_sample[0:validation_start] = \
                    self.y_train[0:validation_start]
                X_train_sample[0:validation_start] = \
                    self.X_train[0:validation_start]

            if i < (folds-1):
                """
                Validation subsample is not located at the end of X.
                """
                if lots_of_info:
                    print("i < (folds-1)")
                    print(f"\t[{validation_start}:{X_train_sample.shape[0]}] len = {X_train_sample.shape[0] - validation_start}")
                    print(f"\t[{validation_stop}:{self.X_train.shape[0]}] len = {self.X_train.shape[0] - validation_stop}")
                
                y_train_sample[validation_start:] = \
                    self.y_train[validation_stop:total_sample_data_points]
                X_train_sample[validation_start:] = \
                    self.X_train[validation_stop:total_sample_data_points]


            inversion_time = time.time()
            self.beta_cv[:, i] = np.linalg.pinv(X_train_sample.T@X_train_sample)@X_train_sample.T@y_train_sample
            inversion_time = time.time() - inversion_time
            if self.timing_info:
                print(f"solved for beta in {inversion_time:.3f} s")

            # Y_predicted[:, i] = X_validation_sample@self.beta_cv[:, i]
            Y_predicted[:, i] = self.X_test@self.beta_cv[:, i]
        
        print(f"X_validation_sample: {X_validation_sample.shape}")
        print(f"beta: {self.beta_cv.shape}")
        print("LOL", Y_validation_sample.shape, Y_predicted.shape)

        # self.beta_cv = np.mean(self.beta_cv, axis=1)

        # y_predicted = self.X_test@self.beta_cv  # Predictions based on the 'holy' test set.
        self.mse_test_cv = mean_squared_error(self.y_test.reshape(-1, 1), Y_predicted)


    def ols(self, lambd):
        pass


class Compare:
    def __init__(self, max_degree, N, noise_factor):
        """
        Parameters
        ----------
        max_degree : int
            Max polynomial degree.

        N : int
            N**2 is the number of data points.

        noise_factor : float, int
            How much noise to add.
        """
        self.degrees = np.arange(1, max_degree+1, 1)
        self.N_degrees = len(self.degrees)
        self.N = N
        self.noise_factor = noise_factor

        self.fig, self.ax = plt.subplots(figsize=(10, 8))


    def compare_no_bootstrap(self, which="mse", show_plot=True):
        """
        Use 'Solve' with a range of polynomial degrees and plot the R score
        or mean squared error as a function of the polynomial degree.

        Parameters
        ----------
        which : string
            Choose whether to plot MSE or R score.  Allowed inputs are 'mse'
            and 'r_score'.  Returns None if any other argument is passed.
        """

        r_score_train = np.empty(self.N_degrees)
        mse_train = np.empty(self.N_degrees)
        r_score_test = np.empty(self.N_degrees)
        mse_test = np.empty(self.N_degrees)

        for i in range(self.N_degrees):
            q = Solve(self.degrees[i], self.N, self.noise_factor,
                debug_info=False, timing_info=True)
            r_score_train[i], mse_train[i], r_score_test[i], mse_test[i] =\
                q.no_bootstrap()


        if which == "mse":
            self.ax.plot(self.degrees, mse_train, label="mse_train")
            self.ax.plot(self.degrees, mse_test, label="mse_test")

        elif which == "r_score":
            self.ax.plot(self.degrees, r_score_train, label="r_score_train")
            self.ax.plot(self.degrees, r_score_test, label="r_score_test")
        
        else:
            print("Please choose 'mse' or 'r_score'.")
            return
        
        self._label_legend_title()
        if show_plot: plt.show()


    def compare_bootstrap(self, n_bootstraps, show_plot=True):
        """
        Compare MSE as a function of polynomial degree.
        """
        mse_test = np.empty(self.N_degrees)
        
        for i in range(self.N_degrees):
            q = Solve(self.degrees[i], self.N, self.noise_factor,
                debug_info=False, timing_info=True)
            q.bootstrap(n_bootstraps)
            
            mse_test[i] = q.mse_test_boot

        self.ax.plot(self.degrees, mse_test, label=f"{n_bootstraps} bootstraps test")
        self._label_legend_title()
        if show_plot: plt.show()


    def compare_bootstrap_bias_variance(self, n_bootstraps, show_plot=True):
        """
        Compare MSE, variance and bias as a function of polynomial
        degree.
        """
        r_score = np.empty(self.N_degrees)
        mse = np.empty(self.N_degrees)
        bias = np.empty(self.N_degrees)
        variance = np.empty(self.N_degrees)
        
        for i in range(self.N_degrees):
            q = Solve(self.degrees[i], self.N, self.noise_factor,
                debug_info=False, timing_info=True)
            q.bootstrap(n_bootstraps)
            r_score[i], mse[i], bias[i], variance[i] = \
                q.r_score_test_boot, q.mse_test_boot, q.bias_test_boot, q.variance_boot
                

        self.ax.semilogy(self.degrees, variance, label="variance")
        self.ax.semilogy(self.degrees, bias, label="bias")
        self.ax.semilogy(self.degrees, mse, label="mse")
        # self.ax.plot(self.degrees, variance+bias, label="var+bias")
        self._label_legend_title()
        if show_plot: plt.show()


    def compare_cross_validation(self, folds=5, show_plot=True):
        mse_test = np.empty(self.N_degrees)
        
        for i in range(self.N_degrees):
            q = Solve(self.degrees[i], self.N, self.noise_factor,
                debug_info=False, timing_info=True)
            q.cross_validation(folds=folds)

            mse_test[i] = q.mse_test_cv


        self.ax.plot(self.degrees, mse_test, label=f"{folds}-fold CV test")
        self._label_legend_title()
        if show_plot: plt.show()


    def _label_legend_title(self):
        self.ax.set_title("LOL")
        self.ax.set_xlabel("Polynomial degree")
        self.ax.set_ylabel("MSE")
        self.ax.legend()


    def compare_cross_validation_folds(self, folds, show_plot=True):

        mse = []
        
        for k in folds:
            q = Solve(deg=self.degrees[-1], N=self.N, noise_factor=self.noise_factor,
                draw_random=False, debug_info=False, timing_info=True)

            mse.append(q.cross_validation(folds=k))

        print(mse)

        self.ax.semilogy(folds, mse)
        self._label_legend_title()
        if show_plot: plt.show()
        

if __name__ == "__main__":
    # np.random.seed(1337)
    # q = Compare(max_degree=8, N=10, noise_factor=0.1)    # Good values for overfitting with no_bootstrap! Don't remove!
    # q.compare_no_bootstrap(which='mse')
    # q.compare_cross_validation(show_plot=True)
    # q.compare_cross_validation_folds(folds=range(1, 15+1))

    # # Task 1a:
    # q = Compare(max_degree=15, N=20, noise_factor=0.1)
    # q.compare_no_bootstrap()

    # # Task 1b:
    # q = Compare(max_degree=12, N=20, noise_factor=0.1)
    # q.compare_bootstrap_bias_variance(n_bootstraps=20)
    
    # # Task 1c:
    # q = Compare(max_degree=30, N=20, noise_factor=0.1)
    # q.compare_bootstrap(n_bootstraps=20, show_plot=True)
    # q.compare_cross_validation(show_plot=True)

    process = psutil.Process()
    print(f"mem. usage: {process.memory_info().rss/1e6:.1f} MB")