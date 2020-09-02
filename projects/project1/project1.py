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
    return np.sum(y_observed - y_predicted)**2/len(y_observed)


def r_squared(y_observed, y_predicted):
    """
    Calculate the score R**2.

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
        The R**2 score.
    """
    return 1 - np.sum(y_observed - y_predicted)**2/\
        np.sum(y_observed - np.mean(y_observed))**2


def create_design_matrix(x1, x2, deg, N):
    """
    Construct a design matrix with N rows and deg+1 columns.

    Parameters
    ----------

    x1 : numpy.ndarray
        Dependent / outcome / response variable. Is it, though?

    x2 : numpy.ndarray
        Dependent / outcome / response variable. Is it, though?

    deg : int
        The polynomial degree.

    N : int
        The number of data points.

    Returns
    -------
    X : numpy.ndarray
        Design matrix of dimensions N rows and deg+1 columns.
    """

    X = np.zeros((N, deg+1))
    X[:, 0] = 1      # Intercept.
    
    for i in range(1, deg+1):
        """
        Runs through all but the 0th column of X.
        """
        for j in range(i+1):
            """
            Runs through all combinations of x and y which produces an
            i'th degree polynomial.
            """
            X[:, i] += (x1**j)*(x2**(i - j))  # THIS IS INCORRECT. EACH TERM SHOULD HAVE INDIVIDUAL COLUMNS!

    return X




def solve():
    np.random.seed(1337)
    N = 10      # Number of data ponts.
    deg = 5     # Polynomial degree.
    x1 = np.random.random(size=N)
    x2 = np.random.random(size=N)

    y_observed = franke_function(x1, x2)
    X = create_design_matrix(x1, x2, deg, N)

    # y_observed += stochastic_noise
    # l = (degree + 2)*(degree + 1)/2

    # X_train, X_test, y_train, y_test = train_test_split()

    # beta = np.linalg.inv( )



if __name__ == "__main__":
    solve()