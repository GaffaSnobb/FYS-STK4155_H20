import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skl
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
        Predicted values.

    Returns
    -------
    : numpy.ndarray
        The mean squared error.
    """
    # return np.sum((y_observed - y_predicted)**2)/len(y_observed)
    return np.mean((y_observed - y_predicted)**2)

def bias(f, y_observed):
    """
    Calculate the bias.

    Parameters
    ----------
    f : numpy.ndarray
        Function values.

    y_observed : numpy.ndarray
        (AKA y_tilde)

    Returns
    -------
    : numpy.ndarray
        The bias.
    """
    
    return mean_squared_error(f, np.mean(y_observed))


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


def create_design_matrix(x1, x2, N, deg):
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
    
    features = int((deg + 1)*(deg + 2)/2)
    X = np.empty((N**2, features))     # Data points x features.
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


def solve(deg, N=34, noise_factor=0.15, debug_info=False, timing_info=False):
    """
    Solve the OLS on the Franke function.

    Draw N numbers in the inverval [0, 1) for both variables, x1 and x2.
    Make a meshgrid of x1 and x2 to ensure all combinations of x1 and x2
    values.  Pass the meshgrids to the Franke function, ravel the
    resulting array for easier calculations and add stochastic noise
    drawn from the standard normal distribution. Create the design
    matrix X based on the meshgrids, the number of randomly drawn points
    and the polynomial degree.  Split the data into training and test
    sets.  Scale the data by subtracting the mean and dividing by the
    standard deviation.  Solve for the vector beta by matrix inversion
    and matrix multiplication.  Use the beta vector to generate model
    data (y_tilde) and predicted data (y_predict). Return the R^2 score
    and MSE for both training and test data sets.

    Parameters
    ----------
    deg : int
        Polynomial degree.

    N : int
        The number of data points per random variable.  The resulting
        meshgrids will measure NxN values.

    noise_factor : int, float
        The factor of added stochastic noise.

    debug_info : boolean
        For toggling print of debug data on / off.

    timing_info : boolean
        For toggling print of timing info on / off.

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
    # x1, x2 = np.meshgrid(np.random.randn(N), np.random.randn(N))
    x1, x2 = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

    y_observed = franke_function(x1, x2).ravel()
    y_observed += noise_factor*np.random.randn(N**2) # Stochastic noise.
    
    create_time = time.time()
    X = create_design_matrix(x1, x2, N, deg)
    create_time = time.time() - create_time

    if timing_info:
        print(f"design matrix created in {create_time:.3f} s")
        print(f"design matrix dimensions {X.shape}")
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y_observed, test_size=0.2)

    # Scaling.
    X_mean = np.mean(X_train)
    X_std = np.std(X_train)
    X_train = (X_train - X_mean)/X_std
    X_test = (X_test - X_mean)/X_std

    betaa = np.linalg.pinv(X_train.T@X_train)@X_train.T@y_train

    # y_tilde = X_train@beta
    # y_predict = X_test@beta

    n_bootstraps = 1
    Y_predict = np.empty((X_test.shape[0], n_bootstraps))
    beta = np.empty((X_test.shape[1], n_bootstraps))

    for i in range(n_bootstraps):
        X_train_resample, y_train_resample = resample(X_train, y_train, replace=True)

        inversion_time = time.time()
        # beta = np.linalg.pinv(X_train_resample.T@X_train_resample)@X_train_resample.T@y_train_resample
        beta[:, i] = np.linalg.pinv(X_train_resample.T@X_train_resample)@X_train_resample.T@y_train_resample
        inversion_time = time.time() - inversion_time

        if timing_info: print(f"solved for beta in {inversion_time:.3f} s")

        Y_predict[:, i] = X_test@beta[:, i]

    # y_tilde = X_train@(np.mean(beta, axis=1))
    y_tilde = X_train@betaa
    y_predict = np.mean(Y_predict, axis=1)

    r_score_train = r_squared(y_train, y_tilde)
    mse_train = mean_squared_error(y_train, y_tilde)
    # bias_train = np.mean( (y_test - np.mean(y_predict))**2 )
    
    
    r_score_test = r_squared(y_test, y_predict)
    mse_test = mean_squared_error(y_test, y_predict)

    if debug_info:
        print("\ntrain")
        print(f"R^2: {r_score_train}")
        print(f"MSE: {mse_train}")
        
        print("train (sklearn)")
        print(f"R^2: {skl.r2_score(y_train, y_tilde)}")
        print(f"MSE: {skl.mean_squared_error(y_train, y_tilde)}")
        
        print("\ntest")
        print(f"R^2: {r_score_test}")
        print(f"MSE: {mse_test}")
        
        print("test (sklearn)")
        print(f"R^2: {skl.r2_score(y_test, y_predict)}")
        print(f"MSE: {skl.mean_squared_error(y_test, y_predict)}")

    return r_score_train, mse_train, r_score_test, mse_test


def compare():
    """
    Call 'solve' with a range of polynomial degrees and plot the R score
    and mean squared error as a function of the polynomial degree.
    """
    # np.random.seed(1337)
    degrees = np.arange(1, 30+1, 1)
    N_degrees = len(degrees)
    N = 40  # The number of rows in X is N**2.
    noise_factor = 0.15
    
    r_score_train = np.empty(N_degrees)
    mse_train = np.empty(N_degrees)
    r_score_test = np.empty(N_degrees)
    mse_test = np.empty(N_degrees)

    for i in range(N_degrees):
        r_score_train[i], mse_train[i], r_score_test[i], mse_test[i] = \
            solve(degrees[i], N, noise_factor, debug_info=False, timing_info=True)

    # plt.plot(degrees, r_score_train, label="r_score_train")
    plt.semilogy(degrees, mse_train, label="mse_train")
    # plt.plot(degrees, r_score_test, label="r_score_test")
    plt.semilogy(degrees, mse_test, label="mse_test")
    plt.xlabel("ploynomial degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


    

if __name__ == "__main__":
    # np.random.seed(1337)
    # solve(deg=2, N=3, debug_info=True)
    compare()