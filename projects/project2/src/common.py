import time
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import Lasso

def polynomial_1d(x, *beta):
    """
    n'th degree polynomial for fit testing.

    Parameters
    ----------
    x : numpy.ndarray
        Dependent variable.

    *beta : floats
        Polynomial coefficients.
    """
    res = np.zeros(len(x))
    for i in range(len(beta)):
        res += beta[i]*x**i

    return res


def mean_squared_error(x, y):
    """
    Calculate the mean squared error.

    Consider adding the length n as an argument if this function is
    called many times.

    Parameters
    ----------
    x : numpy.ndarray
        Value 1.

    y : numpy.ndarray
        Value 2.

    Returns
    -------
    : numpy.ndarray
        The mean squared error.
    """
    return np.mean((x - y)**2)


def r_squared(y_observed, y_predicted):
    """
    Calculate the score R**2.
    
    Parameters
    ----------
    y_observed : numpy.ndarray
        Observed values.
    
    y_predicted : numpy.ndarray
        Predicted values.
    
    Returns
    -------
    : numpy.ndarray
        The R**2 score.
    """
    return 1 - np.sum((y_observed - y_predicted)**2)/\
        np.sum((y_observed - np.mean(y_observed))**2)


def step_length(t, t0, t1):
    return t0/(t + t1)


def franke_function(x, y):
    return 0.75*np.exp(-(9*x - 2)**2/4 - (9*y - 2)**2/4) + \
        0.75*np.exp(-(9*x + 1)**2/49 - (9*y + 1)/10) + \
        0.5*np.exp(-(9*x - 7)**2/4 - (9*y - 3)**2/4) - \
        0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)


def features(degree, n_dependent_variables):
    """
    Calculate the number of features for a given polynomial degree for
    one or two dependent variables.

    Parameters
    ----------
    degree : int
        Polynomial degree.

    n_dependent_variables : int
        The number of dependent variables.
    """
    allowed = [1, 2]
    success = n_dependent_variables in allowed
    msg = "The number of dependent variables must be one of the following:"
    msg += f" {allowed}"
    assert success, msg
    
    if n_dependent_variables == 1:
        return degree + 1
    elif n_dependent_variables == 2:
        return int((degree + 1)*(degree + 2)/2)


def create_design_matrix_two_dependent_variables(x1, x2, N, degree):
    """
    Construct a design matrix with N rows and features =
    (degree + 1)*(degree + 2)/2 columns.  N is the number of samples and
    features is the number of features of the design matrix.  For two
    dependent variables.

    Parameters
    ----------

    x1 : numpy.ndarray
        Dependent variable.

    x2 : numpy.ndarray
        Dependent variable.

    N : int
        The number of data ponts.

    degree : int
        The polynomial degree.

    Returns
    -------
    X : numpy.ndarray
        Design matrix of dimensions N rows and (degree + 1)*(degree + 2)/2
        columns.
    """
    
    X = np.empty((N, features(degree, 2)))     # Data points x features.
    X[:, 0] = 1 # Intercept.
    col_idx = 1 # For indexing the design matrix columns.

    for j in range(1, degree+1):
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


def create_design_matrix_one_dependent_variable(x, n_data_total, poly_degree):
    """
    Construct a design matrix with 'n_data_total' rows and
    'poly_degree+1' columns.  For one dependent variable.

    Parameters
    ----------
    x : numpy.ndarray
        Dependent variable.

    n_data_total : int
        The number of data points (rows).

    poly_degree : int
        The polynomial degree (cols-1).

    Returns
    -------
    X : numpy.ndarray
        Design matrix.
    """
    X = np.empty((n_data_total, features(poly_degree, 1)))
    X[:, 0] = 1 # Intercept.

    for i in range(1, poly_degree+1):
        X[:, i] = x**i

    return X


def ols(X, y, lambd=0):
    """
    Solve for beta using OLS. beta = ((X^TX)^-1)X^Ty.

    Parameters
    ----------
    X : numpy.ndarray
        2D array of X.

    y : numpy.ndarray
        1D array of y.

    lambd : float
        Ridge regression parameter.  Defaults to 0 which means no ridge
        regression.

    Returns
    -------
    beta : numpy.ndarray
        OLS solution.
    """
    return np.linalg.pinv(X.T@X + lambd*np.identity(X.shape[1]))@X.T@y


class _StatTools:
    def __init__(self, n_data_total, poly_degree, init_beta=None, verbose=False):
        """
        Parameters
        ----------
        n_data_total : int
            The total number of data points.

        poly_degree : int
            The polynomial degree.

        init_beta : NoneType, numpy.ndarray
            Initial beta values.  Defaults to None where 0 is used.
        """
        self.n_data_total = n_data_total
        self.poly_degree = poly_degree
        self.n_features = features(self.poly_degree, self.n_dependent_variables)
        self.init_beta = init_beta
        self.verbose = verbose

        self._split_scale()


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
        self.reset_state()    # Reset beta for every new GD.
        if self.verbose: self.start_timing()
        
        for _ in range(iterations):
            """
            Loop over the gradient descents.
            """
            gradient = self.X_train.T@(self.X_train@self.beta - self.y_train)
            gradient *= 2/self.n_data_total
            self.beta -= step_size*gradient

        if self.verbose: self.stop_timing()


    def stochastic_gradient_descent(self, n_epochs, n_batches,
        input_step_size=None, lambd=0):
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

        lambd : float
            Ridge regression penalty parameter.  Defaults to 0 where no
            ridge penalty is applied.

        input_step_size : NoneType, float
            The gradient step size / learning rate.  Defaults to None
            where a dynamic step size is used.
        """
        self.reset_state()    # Reset beta for every new SGD.
        if self.verbose: self.start_timing()
        
        rest = self.n_data_total%n_batches # The rest after equally splitting X into batches.
        n_data_per_batch = self.n_data_total//n_batches # Index step size.
        # Indices of X corresponding to start point of the batches.
        batch_indices = np.arange(0, self.n_data_total-rest, n_data_per_batch)

        momentum_parameter = 0.5
        momentum = 0

        for epoch in range(n_epochs):
            """
            Loop over epochs.
            """
            t_step = epoch*n_data_per_batch   # Does not need to be calculated in the inner loop.
            
            # for i in range(n_data_per_batch):
            for i in range(n_batches):
                """
                Loop over all data in each batch.  For each loop, a
                random start index defined by the number of batches,
                is drawn.  This chooses a random batch by slicing the
                design matrix.
                """
                random_index = np.random.choice(batch_indices)
                X = self.X_train[random_index:random_index+n_data_per_batch]
                y = self.y_train[random_index:random_index+n_data_per_batch]
                t_step += i
                
                if input_step_size is None:
                    step_size = step_length(t=t_step, t0=5, t1=50)
                else: step_size = input_step_size

                gradient = X.T@((X@self.beta) - y)*2/n_data_per_batch
                gradient += 2*lambd*self.beta   # Ridge addition.
                momentum = momentum_parameter*momentum + step_size*gradient
                self.beta -= momentum

        if self.verbose: self.stop_timing()


    def _split_scale(self):
        """
        Split the data into training and test sets.  Scale the data by
        subtracting the mean and dividing by the standard deviation,
        both values from the training set.  Shuffle the values.
        """
        # Splitting.
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, shuffle=True)

        # Scaling.
        # self.X_mean = np.mean(self.X_train)
        # self.X_std = np.std(self.X_train)
        # self.X_train = (self.X_train - self.X_mean)/self.X_std
        # self.X_test = (self.X_test - self.X_mean)/self.X_std


    def reset_state(self):
        """
        Reset beta to the initial guess state and shuffle training data.
        """
        if self.init_beta is None:
            self.beta = np.zeros(self.n_features)
        else:
            msg = "Initial beta value array must be of length"
            msg += f" {self.n_features}, got {len(self.init_beta)}."
            success = len(self.init_beta) == (self.n_features)
            assert success, msg
            
            self.beta = self.init_beta

        state = np.random.get_state()
        np.random.shuffle(self.X_train)
        np.random.set_state(state)
        np.random.shuffle(self.y_train)


    def start_timing(self):
        self.stopwatch = time.time()


    def stop_timing(self):
        self.stopwatch = time.time() - self.stopwatch
        print(f"{sys._getframe().f_back.f_code.co_name} time: {self.stopwatch:.4f} s")

    @property
    def mse(self):
        mse_train = mean_squared_error(self.y_train, self.X_train@self.beta)
        mse_test = mean_squared_error(self.y_test, self.X_test@self.beta)

        return mse_train, mse_test


class Regression:
    using_ridge = False # For suppressing repeated messages.
    using_lasso = False # For suppressing repeated messages.
    def __init__(self, n_data_points, noise_factor, max_poly_degree,
            split_scale=True):
        """
        Create design matrix self.X and observed function values
        self.y_observed based on randomly drawn numbers in [0, 1).

        Parameters
        ----------
        n_data_points : int
            The total number of data points.

        noise_factor : float
            How much noise to add.  The noise is randomly drawn numbers
            from the standard normal distribution with this factor
            multiplied.

        max_poly_degree : int
            The maximum polynomial degree.

        split_scale : boolean
            For toggling split and scaling on / off.
        """

        x1 = np.random.uniform(0, 1, n_data_points)
        x2 = np.random.uniform(0, 1, n_data_points)
        noise = noise_factor*np.random.randn(n_data_points)

        self.y = franke_function(x1, x2) + noise
        self.X = create_design_matrix_two_dependent_variables(x1, x2, n_data_points, max_poly_degree)
        self.n_data_points = n_data_points
        self.max_poly_degree = max_poly_degree
        self.noise_factor = noise_factor
        if split_scale: self._split_scale()

    
    def cross_validation(self, degree, folds, lambd=0, alpha=0,
        shuffle=False):
        """
        Parameters
        ----------
        degree : int
            Ploynomial degree.  The design matrix will be sliced
            according to 'degree' to keep the correct number of
            features (columns).

        folds : int
            The number of folds.

        lambd : float
            Ridge regression parameter.  Defaults to 0 which means no
            ridge regression.

        alpha : float
            Lasso regression parameter.  Defaults to 0 which means no
            Lasso regression.

        shuffle : bool
            Toggle shuffling of design matrix data on / off.  Defaults
            to off.

        Returns
        -------
        mse/folds : float
            The mean squared error of the cross validation.
        """

        if (lambd != 0) and (alpha != 0):
            print("WARNING: Both lambda (Ridge) and alpha (Lasso) are specified."
                + " Alpha will override.")

        if (lambd != 0) and not self.using_ridge:
            print("Using Rigde regression.")
            self.using_ridge = True

        if (alpha != 0) and not self.using_lasso:
            print("Using Lasso regression.")
            self.using_lasso = True

        if degree > self.max_poly_degree:
            print("Input polynomial degree cannot be larger than max_poly_degree.")
            return
        
        X = self.X[:, :features(degree)] # Slice the correct number of features.

        rest = self.n_data_points%folds # The leftover data points which will be excluded.
        X = X[:self.n_data_points-rest] # Remove the rest to get equally sized folds.
        y = self.y[:self.n_data_points-rest] # Remove the rest to get equally sized folds.

        if shuffle:
            # Shuffle data for every new k fold.
            state = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(state)
            np.random.shuffle(y)

        mse = 0
        mse_training = 0
        self.r_score_cv = 0
        
        for i in range(folds):
            """
            Loop over all folds.  Split data sets accordingly.
            """
            y_split = np.split(y, folds)
            y_validation = y_split.pop(i)
            y_training = np.concatenate(y_split)

            X_split = np.split(X, folds)
            X_validation = X_split.pop(i)
            X_training = np.concatenate(X_split)

            X_mean = np.mean(X_training)
            X_std = np.std(X_training)
            X_training = (X_training - X_mean)/X_std
            X_validation = (X_validation - X_mean)/X_std

            if alpha == 0:
                """
                OLS or ridge regression.
                """
                beta = ols(X_training, y_training, lambd)
            else:
                """
                Lasso regression.
                """
                clf = Lasso(alpha=alpha, fit_intercept=False,
                    normalize=True, max_iter=10000, tol=0.07)
                clf.fit(X_training, y_training)
                beta = clf.coef_
            
            y_predicted = X_validation@beta
            y_model = X_training@beta
            mse += mean_squared_error(y_predicted, y_validation)
            mse_training += mean_squared_error(y_model, y_training)

            self.r_score_cv += 1 - np.sum((y_validation - y_predicted)**2)/\
                np.sum((y_validation - np.mean(y_validation))**2)

        self.r_score_cv /= folds

        return mse/folds, mse_training/folds


    def standard_least_squares_regression(self, degree):
        """
        Perform the standard least squares regression (no resampling).

        Parameters
        ----------
        degree : int
            Ploynomial degree.  The design matrix will be sliced
            according to 'degree' to keep the correct number of
            features (columns).

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

        beta : numpy.ndarray
            OLS solution vector.

        var_beta : numpy.ndarray
            The variance of each beta parameter.
        """
        self._split_scale()
        n_features = features(degree)
        X_train = self.X_train[:, :n_features] # Slice the correct number of features.
        X_test = self.X_test[:, :n_features] # Slice the correct number of features.
        
        beta = ols(X_train, self.y_train)
        var_beta = np.diag(np.linalg.pinv(X_test.T@X_test))
        y_model = X_train@beta
        y_predicted = X_test@beta

        r_score_train = r_squared(self.y_train, y_model)
        r_score_test = r_squared(self.y_test, y_predicted)
        mse_train = mean_squared_error(self.y_train, y_model)
        mse_test = mean_squared_error(self.y_test, y_predicted)

        return r_score_train, mse_train, r_score_test, mse_test, beta, \
            var_beta


    def bootstrap(self, degree, n_bootstraps, lambd=0, alpha=0):
        """
        Perform the OLS with bootstrapping.

        Parameters
        ----------
        degree : int
            Ploynomial degree.  The design matrix will be sliced
            according to 'degree' to keep the correct number of
            features (columns).

        n_bootstraps : int
            The number of bootstrap samples.

        lambd : float
            Ridge regression parameter.  Defaults to 0 which means no
            ridge regression.

        alpha : float
            Lasso regression parameter.  Defaults to 0 which means no
            lasso regression.

        Returns
        -------
        mse_boot : float
            The mean squared error of the test set.
        
        variance_boot : float
            The variance.

        bias_boot : float
            The bias.

        alpha : float
            Lasso regression parameter.  Defaults to 0 which means no
            Lasso regression.
        """
        
        # Slicing the correct number of features.
        n_features = features(degree)
        X_train = self.X_train[:, :n_features]
        X_test = self.X_test[:, :n_features]

        # Keep all the predictions for later calculations.
        n_test_data_points = self.X_test.shape[0]
        Y_predicted = np.empty((n_test_data_points, n_bootstraps))
        
        self.r_score_boot = 0

        for b in range(n_bootstraps):
            """
            Draw 'n_bootstrap' bootstrap resamples and calculate
            predicted y values based on every resample.
            """
            X_resample, y_resample = resample(X_train, self.y_train)
            if alpha == 0:
                """
                Use OLS or ridge regression.
                """
                beta = ols(X_resample, y_resample, lambd)
            else:
                """
                Use lasso regression.
                """
                clf = Lasso(alpha=alpha, fit_intercept=False, normalize=True, max_iter=10000, tol=0.07)
                clf.fit(X_resample, y_resample)
                beta = clf.coef_


            Y_predicted[:, b] = X_test@beta

            self.r_score_boot += 1 - np.sum((self.y_test - Y_predicted[:, b])**2)/\
                np.sum((self.y_test - np.mean(self.y_test))**2)

        self.r_score_boot /= n_bootstraps
        mse_boot = mean_squared_error(self.y_test.reshape(-1, 1), Y_predicted)
        variance_boot = np.mean(np.var(Y_predicted, axis=1))
        bias_boot = np.mean((self.y_test - np.mean(Y_predicted, axis=1))**2)
        
        return mse_boot, variance_boot, bias_boot


    def _split_scale(self):
        """
        Split the data into training and test sets.  Scale the data by
        subtracting the mean and dividing by the standard deviation,
        both values from the training set.  Shuffle the values.
        """
        # Splitting.
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, shuffle=True)

        # Scaling.
        X_mean = np.mean(self.X_train)
        X_std = np.std(self.X_train)
        self.X_train = (self.X_train - X_mean)/X_std
        self.X_test = (self.X_test - X_mean)/X_std