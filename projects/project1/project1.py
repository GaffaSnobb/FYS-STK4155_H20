import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def franke_function(x1, x2):
    return 0.75*np.exp(-(0.25*(9*x1 - 2)**2) - 0.25*((9*x2 - 2)**2))\
        + 0.75*np.exp(-((9*x1 + 1)**2)/49.0 - 0.1*(9*x2 + 1))\
        + 0.5*np.exp(-(9*x1 - 7)**2/4.0 - 0.25*((9*x2 - 3)**2))\
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
    Construct a design matrix with N rows and features =
    (deg + 1)*(deg + 2)/2 columns.  N is the number of samples and
    features is the number of features of the design matrix.

    Parameters
    ----------

    x1 : numpy.ndarray
        Dependent / outcome / response variable. Is it, though?

    x2 : numpy.ndarray
        Dependent / outcome / response variable. Is it, though?

    N : int
        The number of data points.

    deg : int
        The polynomial degree.

    Returns
    -------
    X : numpy.ndarray
        Design matrix of dimensions N rows and (deg + 1)*(deg + 2)/2
        columns.
    """

    features = int((deg + 1)*(deg + 2)/2)
    X = np.empty((N, features))     # Samples x features.
    X[:, 0] = 1     # Intercept.
    idx = 1         # For indexing the design matrix.
    
    for i in range(1, deg+1):
        """
        Runs through all polynomial degrees.
        """
        for j in range(i+1):
            """
            Runs through all combinations of x and y which produces an
            i'th degree polynomial.
            """
            X[:, idx] = (x1**j)*(x2**(i - j))
            idx += 1
    
    return X


def solve():
    np.random.seed(1337)
    N = 100      # Number of data ponts.
    deg = 3     # Polynomial degree.
    x1 = np.random.random(size=N)
    x2 = np.random.random(size=N)

    y_observed = franke_function(x1, x2)    # Is observed a good name?
    X = create_design_matrix(x1, x2, N, deg)

    # y_observed += 0.1*np.random.randn(N)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y_observed, test_size=0.2)

    beta = np.linalg.inv(X_train.T@X_train)@X_train.T@y_train

    y_tilde = X_train@beta
    y_predict = X_test@beta

    print("train")
    print(f"R^2: {r_squared(y_train, y_tilde)}")
    print(f"MSE: {mean_squared_error(y_train, y_tilde)}")
    print("test")
    print(f"R^2: {r_squared(y_test, y_predict)}")
    print(f"MSE: {mean_squared_error(y_test, y_predict)}")



if __name__ == "__main__":
    solve()