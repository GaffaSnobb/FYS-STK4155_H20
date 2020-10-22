from math import exp
import common
import numpy as np

class Example2D(common._StatTools):
    def __init__(self, n_data_total, poly_degree, init_beta=None):
        """
        Set up a 2D example using Franke data.

        Parameters
        ----------
        n_data_total : int
            The number of data points.

        poly_degree : int
            The polynomial degree.
        """
        self.n_dependent_variables = 2
        self.x1 = np.random.uniform(0, 1, n_data_total)
        self.x2 = np.random.uniform(0, 1, n_data_total)

        self.X = common.create_design_matrix_two_dependent_variables(self.x1,
            self.x2, n_data_total, poly_degree)
        self.y = common.franke_function(self.x1, self.x2)
        
        super(Example2D, self).__init__(n_data_total, poly_degree, init_beta)

def test_design_matrix_dimensions():
    q = Example2D(2, 5)

    expected = (2, 21)
    success = expected == q.X.shape
    msg = "Design matrix dimensions do not match."
    msg += f" Expected: {expected} got {q.X.shape}."

    assert success, msg

if __name__ == "__main__":
    test_design_matrix_dimensions()