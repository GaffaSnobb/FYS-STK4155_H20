import numpy as np

def franke_function(x, y):
    return 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))\
        + 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))\
        + 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))\
        - 0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)


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


def solve():
    pass


if __name__ == "__main__":
    solve()