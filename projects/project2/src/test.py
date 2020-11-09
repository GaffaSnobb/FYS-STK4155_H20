import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import common

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


class FFNNSingle(common.FFNN):
    """
    Feedforward neural network with one layer hard-coded.
    """
    def __init__(self, X, y, verbose=False):
        """
        Parameters
        ----------
        X : numpy.ndarray
            Design matrix.

        y : numpy.ndarray
            True output.

        verbose : boolean
            Toggle verbose mode on / off.
        """
        self.n_hidden_neurons = 50
        
        super(FFNNSingle, self).__init__(X=X, y=y, verbose=verbose)


    def _initial_state_single(self):
        """
        Set the system to the correct state before training starts.
        Split the data into training and testing sets.  Initialize the
        weights and biases for the hidden layer and the output layer.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, shuffle=True)
        self.y_train = common.to_categorical(self.y_train)

        # Weights and biases for the hidden layer.
        self.hidden_weights = np.random.normal(size=(self.n_features, self.n_hidden_neurons))
        self.hidden_biases = np.full(shape=self.n_hidden_neurons, fill_value=0.01)

        # Weights and biases for the output layer.
        self.output_weights = np.random.normal(size=(self.n_hidden_neurons, self.n_categories))
        self.output_biases = np.full(shape=self.n_categories, fill_value=0.01)


    def _backpropagation_single(self):
        """
        Perform one backpropagation.
        """
        self.output_error = self.probabilities - self.y_selection    # Loss / cost.
        self.hidden_error = self.output_error@self.output_weights.T
        self.hidden_error *= self.a_hidden*(1 - self.a_hidden) # Hard coded Sigmoid derivative.

        output_weights_gradient = self.a_hidden.T@self.output_error
        self.output_bias_gradient = np.sum(self.output_error, axis=0)

        hidden_weights_gradient = self.X_selection.T@self.hidden_error
        hidden_biases_gradient = np.sum(self.hidden_error, axis=0)

        self.output_weights -= self.learning_rate*output_weights_gradient
        self.output_biases -= self.learning_rate*self.output_bias_gradient
        self.hidden_weights -= self.learning_rate*hidden_weights_gradient
        self.hidden_biases -= self.learning_rate*hidden_biases_gradient


    def feedforward_single(self):
        """
        Perform one feedforward.
        """
        self.a_hidden = common.sigmoid(self.X_selection@self.hidden_weights + self.hidden_biases)
        self.z_output = self.a_hidden@self.output_weights + self.output_biases
        z_output_exp = np.exp(self.z_output)
        self.probabilities = z_output_exp/np.sum(z_output_exp, axis=1, keepdims=True)


    def train_neural_network_single(self, learning_rate=0.1, lambd=0):
        """
        Train the neural network.
        """
        self._initial_state_single()
        if self.verbose: self.start_timing()
        self.learning_rate = learning_rate
        self.lambd = lambd

        data_indices = np.arange(self.X_train.shape[0])
        n_iterations = self.n_data_total//self.batch_size

        for _ in range(self.n_epochs):
            """
            Loop over epochs.
            """
            for _ in range(n_iterations):
                """
                Loop over iterations.  The number of iterations is the
                total number of data points divided by the batch size.
                """
                minibatch_indices = np.random.choice(data_indices,
                    size=self.batch_size, replace=True)

                self.X_selection = self.X_train[minibatch_indices]
                self.y_selection = self.y_train[minibatch_indices]

                self.feedforward_single()
                self._backpropagation_single()

        if self.verbose: self.stop_timing()


def test_design_matrix_dimensions():
    q = Example2D(2, 5)

    expected = (2, 21)
    success = expected == q.X.shape
    msg = "Design matrix dimensions do not match."
    msg += f" Expected: {expected} got {q.X.shape}."

    assert success, msg

tol = 1e-10
digits = datasets.load_digits()

np.random.seed(1337)
q1 = common.FFNN(X=digits.images, y=digits.target)
q1._initial_state()
q1.X_selection = q1.X_train
q1.feedforward()

np.random.seed(1337)
q2 = FFNNSingle(X=digits.images, y=digits.target)
q2._initial_state_single()
q2.X_selection = q2.X_train
q2.feedforward_single()


def test_initial_state_and_feedforward_output_weights():
    for i in range(50):
        for j in range(10):
            if np.abs(q2.output_weights[i, j] - q1.output_weights[i, j]) > tol:
                msg = f"output weight error at row: {i} col: {j}"
                assert False, msg


def test_initial_state_and_feedforward_output_biases():
    for i in range(10):
        if np.abs(q2.output_biases[i] - q1.output_biases[i]) > tol:
            msg = f"output bias error at index: {i}"
            assert False, msg


def test_initial_state_and_feedforward_hidden_weights():
    for i in range(64):
        for j in range(50):
            if np.abs(q2.hidden_weights[i, j] - q1.hidden_weights[0][i, j]) > tol:
                msg = f"hidden weight error at row: {i} col: {j}"
                assert False, msg


def test_initial_state_and_feedforward_hidden_biases():
    for i in range(50):
        if np.abs(q2.hidden_biases[i] - q1.hidden_biases[0][i]) > tol:
            msg = f"hidden bias error at index: {i}"
            assert False, msg


def test_initial_state_and_feedforward_hidden_neuron_input():
    for i in range(360):
        for j in range(50):
            if np.abs(q2.a_hidden[i, j] - q1.neuron_input[1][i, j]) > tol:
                msg = f"neuron input error at row: {i} col: {j}"
                assert False, msg


def test_initial_state_and_feedforward_output_neuron_activation():
    for i in range(360):
        for j in range(10):
            if np.abs(q2.z_output[i, j] - q1.neuron_activation[-1][i, j]) > tol:
                msg = f"neuron activation error at row: {i} col: {j}"
                assert False, msg


np.random.seed(1337)
q3 = common.FFNN(X=digits.images, y=digits.target)
q3._initial_state()
q3.X_selection = q3.X_train
q3.y_selection = q3.y_train
q3.learning_rate = 0.1
q3.lambd = 0
q3.feedforward()
q3._backpropagation()

np.random.seed(1337)
q4 = FFNNSingle(X=digits.images, y=digits.target)
q4._initial_state_single()
q4.X_selection = q4.X_train
q4.y_selection = q4.y_train
q4.learning_rate = 0.1
q4.lambd = 0
q4.feedforward_single()
q4._backpropagation_single()

def test_backpropagation_probabilities():
    for i in range(1437):
        for j in range(10):
            if np.abs(q4.probabilities[i, j] - q3.neuron_input[-1][i, j]) > tol:
                msg = f"probability error at row: {i} col: {j}"
                assert False, msg


def test_backpropagation_output_error():
    for i in range(1437):
        for j in range(10):
            if np.abs(q4.output_error[i, j] - q3.error[-1][i, j]) > tol:
                msg = f"error output error at row: {i} col: {j}"
                assert False, msg


def test_backpropagation_output_bias_gradient():
    for i in range(10):
        if np.abs(q4.output_bias_gradient[i] - q3.bias_gradient[-1][i]) > tol:
            msg = f"output bias gradient error at index {i}"
            assert False, msg


def test_backpropagation_output_biases():
    for i in range(10):
        if np.abs(q4.output_biases[i] - q3.output_biases[i]) > tol:
            msg = f"output bias error at index {i}"
            assert False, msg


def test_backpropagation_hidden_error():
    q3_hidden_error = q3.error[-2].T    # Transpose only once.
    for i in range(1437):
        for j in range(50):
            if np.abs(q4.hidden_error[i, j] - q3_hidden_error[i, j]) > tol:
                msg = f"hidden error error at row: {i} col: {j}"
                assert False, msg


def test_backpropagation_output_weights():
    for i in range(50):
        for j in range(10):
            if np.abs(q4.output_weights[i, j] - q3.output_weights[i, j]) > tol:
                msg = f"output weights error at row: {i} col: {j}"
                assert False, msg


def test_backpropagation_hidden_biases():
    for i in range(50):
        if np.abs(q4.hidden_biases[i] - q3.hidden_biases[0][i]) > tol:
            msg = f"hidden bias error at index: {i}"
            assert False, msg


def test_backpropagation_hidden_weights():
    for i in range(64):
        for j in range(50):
            if np.abs(q4.hidden_weights[i, j] - q3.hidden_weights[0][i, j]) > tol:
                msg = f"hidden weight error at row: {i} col: {j}"
                assert False, msg


if __name__ == "__main__":
    test_design_matrix_dimensions()
    test_initial_state_and_feedforward_output_weights()
    test_initial_state_and_feedforward_output_biases()
    test_initial_state_and_feedforward_hidden_weights()
    test_initial_state_and_feedforward_hidden_biases()
    test_initial_state_and_feedforward_hidden_neuron_input()
    test_initial_state_and_feedforward_output_neuron_activation()
    
    test_backpropagation_probabilities()
    test_backpropagation_output_error()
    test_backpropagation_output_bias_gradient()
    test_backpropagation_output_biases()
    test_backpropagation_hidden_error()
    test_backpropagation_output_weights()
    test_backpropagation_hidden_biases()
    test_backpropagation_hidden_weights