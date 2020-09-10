import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    y_predicted : numpy.ndarray
        Predicted values.

    Returns
    -------
    : numpy.ndarray
        The mean squared error.
    """
    return np.sum((y_observed - y_predicted)**2)/len(y_observed)


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


def solve(debug=False):
    # np.random.seed(1337)
    N = 1000     # Number of randomly drawn data ponts per variable.
    deg = 5    # Polynomial degree.
    x1, x2 = np.meshgrid(np.random.random(size=N), np.random.random(size=N))

    # x1 = np.random.rand(N**2)
    # x2 = np.random.rand(N**2)

    y_observed = franke_function(x1, x2).ravel()
    y_observed += 0.1*np.random.randn(N**2) # Stochastic noise.
    
    create_time = time.time()
    X = create_design_matrix(x1, x2, N, deg)
    create_time = time.time() - create_time
    print(f"design matrix created in {create_time:.3f} s")
    print(f"design matrix dimensions {X.shape}")
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y_observed, test_size=0.2)

    # Scaling.
    X_mean = np.mean(X_train)
    X_std = np.std(X_train)
    X_train = (X_train - X_mean)/X_std
    X_test = (X_test - X_mean)/X_std

    inversion_time = time.time()
    beta = np.linalg.pinv(X_train.T@X_train)@X_train.T@y_train
    inversion_time = time.time() - inversion_time
    print(f"solved for beta in {inversion_time:.3f} s")

    y_tilde = X_train@beta
    y_predict = X_test@beta

    if debug:
        print("\ntrain")
        print(f"R^2: {r_squared(y_train, y_tilde)}")
        print(f"MSE: {mean_squared_error(y_train, y_tilde)}")
        
        print("train (sklearn)")
        print(f"R^2: {skl.r2_score(y_train, y_tilde)}")
        print(f"MSE: {skl.mean_squared_error(y_train, y_tilde)}")
        
        print("\ntest")
        print(f"R^2: {r_squared(y_test, y_predict)}")
        print(f"MSE: {mean_squared_error(y_test, y_predict)}")
        
        print("test (sklearn)")
        print(f"R^2: {skl.r2_score(y_test, y_predict)}")
        print(f"MSE: {skl.mean_squared_error(y_test, y_predict)}")

def compare():
    pass

if __name__ == "__main__":
    solve(debug=True)
    # compare()