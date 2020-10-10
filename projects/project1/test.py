from common import Regression, ols
import numpy as np

class RegressionTest(Regression):
    def __init__(self, shape, fill_value):
        self.X = np.identity(shape)
        self.y = np.full(shape, fill_value)

def test_ols():
    """
    Test that ols returns expected result for identity matrix as design
    matrix and a given set of y values.
    """
    shape = 5
    fill_value = 50
    q = RegressionTest(shape, fill_value)
    expected = np.full(shape, fill_value)
    
    success = np.all(expected == ols(q.X, q.y))
    msg = "OLS alters the input data in an unexpected way!"
    msg += " Check it before usage."
    assert success, msg


if __name__ == "__main__":
    test_ols()