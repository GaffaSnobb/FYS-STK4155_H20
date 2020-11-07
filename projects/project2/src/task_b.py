from tokenize import PseudoExtras
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from common import _StatTools


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def dsigmoid(x):
    """
    Derivative of the sigmoid function.
    """
    val = np.exp(-x)
    return val/(val + 1)**2


def cost(y_predicted, y_actual):
    return y_predicted - y_actual


class FFNN(_StatTools):
    """
    Class implementation of a feedforward neural network.
    """
    def __init__(self, verbose=False):
        """
        verbose : bool
            Toggle verbose mode on / off.
        """
        digits = datasets.load_digits()
        self.X = digits.images
        self.y = digits.target
        self.n_data_total = self.X.shape[0] # Total number of data points.
        self.X = self.X.reshape(self.n_data_total, -1)
        
        self.n_features = self.X.shape[1]   # The number of features.
        self.n_hidden_neurons = 50  # NOT IN USE WITH MULTI LAYER
        self.n_epochs = 50
        self.batch_size = 20        # Size of each minibatch.
        self.n_categories = 10      # Number of output categories. 0, 1, 2, ...

        # self.hidden_layer_sizes = (50, 50, 50)
        self.hidden_layer_sizes = (50,)
        self.n_hidden_layers = len(self.hidden_layer_sizes)
        self.verbose = verbose


    def feedforward(self):
        """
        Perform one feedforward.
        """
        self.neuron_input = np.zeros(shape=self.n_hidden_layers + 2, dtype=np.ndarray)  # a
        self.neuron_activation = np.zeros(shape=self.n_hidden_layers + 2, dtype=np.ndarray) # z

        self.neuron_input[0] = self.X_minibatch # Input to first layer is the design matrix.
        self.neuron_activation[0] = np.array([0])

        for i in range(self.n_hidden_layers):
            """
            Loop over the hidden layers.  Calculate the neuron
            activation and neuron input for all neurons in all hidden
            layers.
            """
            self.neuron_activation[i + 1] = self.neuron_input[i]@self.hidden_weights[i] + self.hidden_biases[i]   # No expontential?
            self.neuron_input[i + 1] = sigmoid(self.neuron_activation[i + 1])

        self.neuron_activation[-1] = self.neuron_input[-2]@self.output_weights + self.output_biases
        self.neuron_input[-1] = sigmoid(self.neuron_activation[-1])

    
    def _backpropagation(self):
        error = np.zeros(shape=self.n_hidden_layers + 1, dtype=np.ndarray)  # Store error for hidden layers and output layer (or is it input?).
        error[-1] = cost(self.neuron_input[-1], self.y_minibatch)*dsigmoid(self.neuron_activation[-1])
        error[-2] = self.output_weights@error[-1].T*dsigmoid(self.neuron_activation[-2]).T

        self.bias_gradient = np.zeros(shape=self.n_hidden_layers + 1, dtype=np.ndarray)
        self.bias_gradient[-1] = np.sum(error[-1], axis=0)  # Why axis 0 here?
        self.bias_gradient[-2] = np.sum(error[-2], axis=1)  # Why axis 1 here?

        self.weight_gradient = np.zeros(shape=self.n_hidden_layers + 1, dtype=np.ndarray)
        self.weight_gradient[-1] = (error[-1].T@self.neuron_input[-2]).T # SHOULD THERE BE A TRANSPOSE HERE? Must be for dims. to match.
        self.weight_gradient[-2] = (error[-2]@self.neuron_input[-3])
        
        for i in range(-3, -self.n_hidden_layers - 2, -1):
            """
            Loop backwards through the errors, bias and weight
            gradients.
            """
            error[i] = self.hidden_weights[i + 2]@error[i + 1]*dsigmoid(self.neuron_activation[i].T)
            self.bias_gradient[i] = np.sum(error[i], axis=1)
            self.weight_gradient[i] = error[i]@self.neuron_input[i - 1]

        self.output_weights -= self.learning_rate*(self.weight_gradient[-1]) + self.lambd*self.output_weights
        self.output_biases -= self.learning_rate*(self.bias_gradient[-1])

        for i in range(-1, -self.n_hidden_layers - 1, -1):
            """
            Loop backwards through the hidden weights and biases.
            """
            self.hidden_weights[i] -= self.learning_rate*(self.weight_gradient[i - 1].T) + self.lambd*(self.hidden_weights[i])
            self.hidden_biases[i] -= self.learning_rate*(self.bias_gradient[i - 1])


    def _initial_state(self):
        """
        Set the system to the correct state before training starts.
        Split the data into training and testing sets.  Initialize the
        weights and biases for the hidden layer(s) and the output layer.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, shuffle=True)
        self.y_train = to_categorical(self.y_train)

        self.hidden_weights = []
        self.hidden_biases = []

        # Special case for the first hidden layer.
        hidden_weights_tmp = np.random.normal(size=(self.n_features, self.hidden_layer_sizes[0]))
        self.hidden_weights.append(hidden_weights_tmp)
        hidden_biases_tmp = np.full(shape=self.hidden_layer_sizes[0], fill_value=0.01)
        self.hidden_biases.append(hidden_biases_tmp)

        for i in range(1, self.n_hidden_layers):
            """
            Initialize weights and biases for all hidden layers except
            the first, which was handled before the loop as a special
            case.  The number of rows in the i'th layers hidden weights
            is equal to the number of neurons in the i-1'th layer.
            """
            hidden_weights_tmp = np.random.normal(size=(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i]))
            self.hidden_weights.append(hidden_weights_tmp)
            hidden_biases_tmp = np.full(shape=self.hidden_layer_sizes[i], fill_value=0.01)
            self.hidden_biases.append(hidden_biases_tmp)

        # Weights and biases for the output layer.
        self.output_weights = np.random.normal(size=(self.hidden_layer_sizes[-1], self.n_categories))
        self.output_biases = np.full(shape=self.n_categories, fill_value=0.01)

        # # Weights and biases for the hidden layers.
        # self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        # self.hidden_biases = np.zeros(self.n_hidden_neurons) + 0.01

        # # Weights and biases for the output layer.
        # self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        # self.output_biases = np.zeros(self.n_categories) + 0.01


    def train_neural_network(self, learning_rate=0.1, lambd=0):
        """
        Train the neural network.
        """
        self._initial_state()
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

                self.feedforward()
                # self.feedforward_single()
                self._backpropagation()
                # self._backpropagation_single()
            #     break
            # break

        if self.verbose: self.stop_timing()


    def predict(self, X):
        self.X_minibatch = X
        self.feedforward()
        # print(self.y_test)
        # print(np.sum(self.neuron_input[-1][0]))
        probabilities = self.neuron_activation[-1]/np.sum(self.neuron_activation[-1], axis=1, keepdims=True)
        # print(np.sum(probabilities[-1]))
        # print(np.argmax(probabilities, axis=1))
        # print(self.y_test)
        score = accuracy_score(np.argmax(probabilities, axis=1), self.y_test)
        # print(score)
        return score




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
        self.y_train = to_categorical(self.y_train)

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
        z_output = np.exp(self.a_hidden@self.output_weights + self.output_biases)
        self.probabilities = z_output/np.sum(z_output, axis=1, keepdims=True)


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



if __name__ == "__main__":
    np.random.seed(1337)
    q1 = FFNN()
    q1.train_neural_network()
    score = q1.predict(q1.X_test)
    print(score)
    
    np.random.seed(1337)
    q2 = FFNNSingle()
    q2.train_neural_network_single(learning_rate=0.007)
    score = q2.predict_single(q2.X_test)
    print(score)
    
    # for learning_rate in np.logspace(-5, 0, 8):
    #     q2.train_neural_network_single(learning_rate)
    #     score = q2.predict_single(q2.X_test)
    #     print(f"score: {score} for learning rate: {learning_rate}")