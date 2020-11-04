import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from common import _StatTools


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dsigmoid(x):
    return np.exp(-x)/(np.exp(-x) + 1)**2


class FFNN(_StatTools):
    """
    Class implementation of a feedforward neural network.
    """
    def __init__(self):
        digits = datasets.load_digits()
        self.X = digits.images
        self.y = digits.target
        self.n_data_total = self.X.shape[0] # Total number of data points.
        self.X = self.X.reshape(self.n_data_total, -1)
        
        self.n_features = self.X.shape[1]   # The number of features.
        self.n_hidden_neurons = 50
        self.epochs = 50
        self.batch_size = 20        # Size of each minibatch.
        self.n_categories = 10      # Number of output categories. 0, 1, 2, ...
        # self.learning_rate = 0.1    # Aka eta.
        # self.lambd = 0


    def feedforward(self):
        """
        Perform one feedforward. a_hidden is a weighted sum of inputs
        for the hidden layer with added bias.  z_output is a weighted
        sum of inputs for the output layer with added bias.
        """
        self.a_hidden = sigmoid(self.X_minibatch@self.hidden_weights + self.hidden_biases)
        z_output = np.exp(self.a_hidden@self.output_weights + self.output_biases)
        self.probabilities = z_output/np.sum(z_output, axis=1, keepdims=True)

    
    def _backpropagation(self):
        error_output = self.probabilities - self.y_minibatch    # Loss.
        error_hidden = error_output@self.output_weights.T*self.a_hidden*(1 - self.a_hidden) # Hard coded Sigmoid derivative?

        output_weights_gradient = self.a_hidden.T@error_output
        output_bias_gradient = np.sum(error_output, axis=0)

        hidden_weights_gradient = self.X_minibatch.T@error_hidden
        hidden_biases_gradient = np.sum(error_hidden, axis=0)

        if self.lambd > 0:
            """
            Whats happening here?  Regularization.
            """
            output_weights_gradient += self.lambd*self.output_weights
            hidden_weights_gradient += self.lambd*self.hidden_weights

        self.output_weights -= self.learning_rate*output_weights_gradient
        self.output_biases -= self.learning_rate*output_bias_gradient
        self.hidden_weights -= self.learning_rate*hidden_weights_gradient
        self.hidden_biases -= self.learning_rate*hidden_biases_gradient


    def _initial_state(self):
        """
        Set the system to the correct state before training starts.
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


    def train_neural_network(self, learning_rate=0.1, lambd=0):
        """
        Train the neural network.
        """
        self._initial_state()
        self.start_timing()
        self.learning_rate = learning_rate
        self.lambd = lambd
        
        data_indices = np.arange(self.X_train.shape[0])
        iterations = self.n_data_total//self.batch_size

        for _ in range(self.epochs):
            """
            Loop over epochs.
            """
            for _ in range(iterations):
                """
                Loop over iterations.  The number of iterations is the
                total number of data points divided by the batch size.
                """
                minibatch_indices = np.random.choice(data_indices,
                    size=self.batch_size, replace=True)

                self.X_minibatch = self.X_train[minibatch_indices]
                self.y_minibatch = self.y_train[minibatch_indices]

                self.feedforward()
                self._backpropagation()

        self.stop_timing()


    def predict(self, X):
        self.X_minibatch = X
        self.feedforward()
        # print(np.argmax(self.probabilities, axis=1))
        # print(self.probabilities.shape)
        # print(np.argmax(self.y_test, axis=1))
        # print(self.y_test.shape)
        score = accuracy_score(np.argmax(self.probabilities, axis=1), self.y_test)
        # print(f"score: {score}")
        return score

if __name__ == "__main__":
    np.random.seed(1337)
    q = FFNN()

    learning_rates = np.logspace(-5, 0, 8)

    for learning_rate in learning_rates:

        q.train_neural_network(learning_rate)
        score = q.predict(q.X_test)
        print(f"score: {score} for learning rate: {learning_rate}")