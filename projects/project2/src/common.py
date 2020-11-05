import time
import sys
import numpy as np
from sklearn.model_selection import train_test_split

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

        verbose : bool
            Toggle verbose mode on / off.
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