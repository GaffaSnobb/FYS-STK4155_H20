import numpy as np

def create_design_matrix(x1, x2, N, deg):
    """
    Construct a design matrix with N rows and features =
    (deg + 1)*(deg + 2)/2 columns.  N is the number of samples and
    features is the number of features of the design matrix.

    Parameters
    ----------

    x1 : numpy.ndarray
        Dependent variable.

    x2 : numpy.ndarray
        Dependent variable.

    N : int
        The number of randomly drawn data ponts.

    deg : int
        The polynomial degree.

    Returns
    -------
    X : numpy.ndarray
        Design matrix of dimensions N rows and (deg + 1)*(deg + 2)/2
        columns.
    """
    
    features = int((deg + 1)*(deg + 2)/2)
    X = np.empty((N, features))     # Data points x features.
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