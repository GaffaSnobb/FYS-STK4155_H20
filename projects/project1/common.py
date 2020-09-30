import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def create_design_matrix(x1, x2, N, degree):
    """
    Construct a design matrix with N rows and features =
    (degree + 1)*(degree + 2)/2 columns.  N is the number of samples and
    features is the number of features of the design matrix.

    Parameters
    ----------

    x1 : numpy.ndarray
        Dependent variable.

    x2 : numpy.ndarray
        Dependent variable.

    N : int
        The number of randomly drawn data ponts.

    degree : int
        The polynomial degree.

    Returns
    -------
    X : numpy.ndarray
        Design matrix of dimensions N rows and (degree + 1)*(degree + 2)/2
        columns.
    """
    
    # features = int((degree + 1)*(degree + 2)/2)
    X = np.empty((N, features(degree)))     # Data points x features.
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


def franke_function(x, y):
    return 0.75*np.exp(-(9*x - 2)**2/4 - (9*y - 2)**2/4) + \
        0.75*np.exp(-(9*x + 1)**2/49 - (9*y + 1)/10) + \
        0.5*np.exp(-(9*x - 7)**2/4 - (9*y - 3)**2/4) - \
        0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)


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


def features(degree):
    """
    Calculate the number of features for a given polynomial degree.

    Parameters
    ----------
    degree : int
        Polynomial degree.
    """
    return int((degree + 1)*(degree + 2)/2)


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


class Regression:
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
        self.X = create_design_matrix(x1, x2, n_data_points, max_poly_degree)
        self.n_data_points = n_data_points
        self.max_poly_degree = max_poly_degree
        self.noise_factor = noise_factor
        if split_scale: self._split_scale()

    
    def cross_validation(self, degree, folds, lambd=0):
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
        """

        if degree > self.max_poly_degree:
            print("Input polynomial degree cannot be larger than max_poly_degree.")
            return
        
        X = self.X[:, :features(degree)] # Slice the correct number of features.

        rest = self.n_data_points%folds # The leftover data points which will be excluded.
        X = X[:self.n_data_points-rest] # Remove the rest to get equally sized folds.
        y = self.y[:self.n_data_points-rest] # Remove the rest to get equally sized folds.

        mse = 0
        
        for i in range(folds):
            """
            Loop over all folds.  Split data sets accordingly
            """
            y_split = np.split(y, folds)
            y_validation = y_split.pop(i)
            y_training = np.concatenate(y_split)

            X_split = np.split(X, folds)
            X_validation = X_split.pop(i)
            X_training = np.concatenate(X_split)

            beta = ols(X_training, y_training, lambd)
            y_predicted = X_validation@beta
            mse += mean_squared_error(y_predicted, y_validation)

        return mse/folds


    def standard_least_squares_regression(self, degree):
        """
        Perform the standard least squares regression.

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
        """
        self._split_scale()
        X_train = self.X_train[:, :features(degree)] # Slice the correct number of features.
        X_test = self.X_test[:, :features(degree)] # Slice the correct number of features.
        
        beta = ols(X_train, self.y_train)
        var_beta = self.noise_factor*np.diag(np.linalg.pinv(X_test.T@X_test))
        y_model = X_train@beta
        y_predicted = X_test@beta

        r_score_train = r_squared(self.y_train, y_model)
        r_score_test = r_squared(self.y_test, y_predicted)
        mse_train = mean_squared_error(self.y_train, y_model)
        mse_test = mean_squared_error(self.y_test, y_predicted)

        return r_score_train, mse_train, r_score_test, mse_test, beta, var_beta
        

    def bootstrap(self, degree, n_bootstraps, lambd=0):
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

        Returns
        -------
        mse_boot : float
            The mean squared error of the test set.
        
        variance_boot : float
            The variance.

        bias_boot : float
            The bias.
        """
        
        # For slicing the correct number of features.
        n_features = features(degree)
        X_train = self.X_train[:, :n_features]
        X_test = self.X_test[:, :n_features]

        # Keep all the predictions for later calculations.
        n_test_data_points = self.X_test.shape[0]
        Y_predicted = np.empty((n_test_data_points, n_bootstraps))


        for b in range(n_bootstraps):
            """
            Draw 'n_bootstrap' bootstrap resamples and calculate
            predicted y values based on every resample.
            """
            X_resample, y_resample = resample(X_train, self.y_train)
            beta = ols(X_resample, y_resample, lambd)

            Y_predicted[:, b] = X_test@beta

        mse_boot = mean_squared_error(self.y_test.reshape(-1, 1), Y_predicted)
        variance_boot = np.mean(np.var(Y_predicted, axis=1))
        bias_boot = np.mean((self.y_test - np.mean(Y_predicted, axis=1))**2)
        return mse_boot, variance_boot, bias_boot


    def _split_scale(self):
        """
        Split the data into training and test sets.  Scale the data by
        subtracting the mean and dividing by the standard deviation,
        both values from the training set.
        """
        # Splitting.
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2)

        # Scaling.
        X_mean = np.mean(self.X_train)
        X_std = np.std(self.X_train)
        self.X_train = (self.X_train - X_mean)/X_std
        self.X_test = (self.X_test - X_mean)/X_std

