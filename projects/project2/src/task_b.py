import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

class FFNN:
    """
    Class implementation of a feedforward neural network.
    """
    def __init__(self):
        self.X = np.zeros((2, 2))
        self.y = np.zeros((2, 2))
        
        self.n_hidden_neurons = 5
        self.n_data_total = self.X.shape[0] # Total number of data points.
        self.n_features = self.X.shape[1]   # The number of features.
        self.epochs = 5
        self.batch_size = 5     # Size of each minibatch.
        self.n_categories = 10  # Explain?
        self.learning_rate = 0.1    # Aka eta.
        self.lambd = 0
        
        # Weights and biases for the hidden layers.
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_biases = np.zeros(self.n_hidden_neurons) + 0.01

        # Weights and biases for the output layer.
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_biases = np.zeros(self.n_categories) + 0.01


    def feedforward(self):
        """
        Perform one feedforward step.
        """
        # a_hidden is a weighted sum of inputs for the hidden layer with added bias.
        # z_output is a weighted sum of inputs for the output layer with added bias.
        self.a_hidden = sigmoid(self.X_minibatch@self.hidden_weights + self.hidden_biases)
        z_output = np.exp(np.matmul(self.a_hidden, self.output_weights) + self.output_biases)
        self.probabilities = z_output/np.sum(z_output, axis=1, keepdims=True)

    
    def backpropagation(self):
        error_output = self.probabilities - self.y_minibatch
        error_hidden = np.matmul(error_output, self.output_weights.T)*self.a_hidden*(1 - self.a_hidden)

        output_weights_gradient = np.matmul(self.a_hidden.T, error_output)
        output_bias_gradient = np.sum(error_output, axis=0)

        hidden_weights_gradient = np.matmul(self.X_minibatch.T, error_hidden)
        hidden_biases_gradient = np.sum(error_hidden, axis=0)

        if self.lambd > 0:
            output_weights_gradient += self.lambd*self.output_weights
            hidden_weights_gradient += self.lambd*self.hidden_weights

        self.output_weights -= self.learning_rate*output_weights_gradient
        self.output_biases -= self.learning_rate*output_bias_gradient
        self.hidden_weights -= self.learning_rate*hidden_weights_gradient
        self.hidden_biases -= self.learning_rate*hidden_biases_gradient


    def train(self):
        """
        Train the neural network.
        """
        data_indices = np.arange(self.n_data_total)
        iterations = self.n_data_total // self.batch_size

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

                self.X_minibatch = self.X[minibatch_indices]
                self.y_minibatch = self.y[minibatch_indices]

                self.feedforward()
                self.backpropagation()


if __name__ == "__main__":
    np.random.seed(1337)
    q = FFNN()
    q.X_minibatch = q.X

    q.feedforward()
    print(q.probabilities)