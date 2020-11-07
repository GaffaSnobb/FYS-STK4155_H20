import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import common
from task_b import FFNN, sigmoid

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



class FFNNSingle(FFNN):
    def predict_single(self, X):
        self.X_minibatch = X
        self.feedforward_single()
        score = accuracy_score(np.argmax(self.probabilities, axis=1), self.y_test)
        # print(np.argmax(self.probabilities, axis=1).shape)
        # print(np.argmax(self.probabilities, axis=1))
        return score


    def _initial_state_single(self):
        """
        Remove when multilayer works.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, shuffle=True)
        self.y_train = common.to_categorical(self.y_train)

        # Weights and biases for the hidden layers.
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_biases = np.zeros(self.n_hidden_neurons) + 0.01

        # Weights and biases for the output layer.
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_biases = np.zeros(self.n_categories) + 0.01


    def _backpropagation_single(self):
        """
        Remove when multilayer works.
        """
        error_output = self.probabilities - self.y_minibatch    # Loss.
        error_hidden = error_output@self.output_weights.T*self.a_hidden*(1 - self.a_hidden) # Hard coded Sigmoid derivative?

        output_weights_gradient = self.a_hidden.T@error_output
        output_bias_gradient = np.sum(error_output, axis=0)

        hidden_weights_gradient = self.X_minibatch.T@error_hidden
        hidden_biases_gradient = np.sum(error_hidden, axis=0)

        if self.lambd > 0:
            """
            Regularization.
            """
            output_weights_gradient += self.lambd*self.output_weights
            hidden_weights_gradient += self.lambd*self.hidden_weights

        self.output_weights -= self.learning_rate*output_weights_gradient
        self.output_biases -= self.learning_rate*output_bias_gradient
        self.hidden_weights -= self.learning_rate*hidden_weights_gradient
        self.hidden_biases -= self.learning_rate*hidden_biases_gradient


    def feedforward_single(self):
        """
        Remove when multilayer works.
        """
        self.a_hidden = sigmoid(self.X_minibatch@self.hidden_weights + self.hidden_biases)
        self.z_output = np.exp(self.a_hidden@self.output_weights + self.output_biases)
        self.probabilities = self.z_output/np.sum(self.z_output, axis=1, keepdims=True)


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

                self.X_minibatch = self.X_train[minibatch_indices]
                self.y_minibatch = self.y_train[minibatch_indices]

                self.feedforward_single()
                self._backpropagation_single()
            #     break
            # break

        if self.verbose: self.stop_timing()


def test_multilayer_with_one_layer():
    np.random.seed(1337)
    q1 = FFNN()

    q1._initial_state()
    # print(q1.hidden_biases)
    # print(q1.hidden_weights)
    # print(q1.output_biases)
    # print(type(q1.output_weights))

    q1.X_minibatch = q1.X_test
    q1.feedforward()
    # print(q1.neuron_input[1].shape)
    print(q1.neuron_activation[-1].shape)

    # q1.train_neural_network()
    # score = q1.predict(q1.X_test)
    # print(score)

    np.random.seed(1337)
    q2 = FFNNSingle()

    q2._initial_state_single()
    # print(q2.hidden_biases)
    # print(q2.hidden_weights)
    # print(q2.output_biases)
    # print(type(q2.output_weights))

    q2.X_minibatch = q1.X_test
    q2.feedforward_single()
    # print(q2.a_hidden.shape)
    print(q2.z_output.shape)

    for i in range(50):
        for j in range(10):
            if q2.output_weights[i, j] != q1.output_weights[i, j]:
                msg = f"output weight error at row: {i} col: {j}"
                assert False, msg
            
    for i in range(360):
        for j in range(50):
            if q2.a_hidden[i, j] != q1.neuron_input[1][i, j]:
                msg = f"neuron input error at row: {i} col: {j}"
                assert False, msg

    for i in range(360):
        for j in range(10):
            if q2.z_output[i, j] != q1.neuron_activation[-1][i, j]:
                msg = f"neuron activation error at row: {i} col: {j}"
                assert False, msg


if __name__ == "__main__":
    test_design_matrix_dimensions()
    test_multilayer_with_one_layer()