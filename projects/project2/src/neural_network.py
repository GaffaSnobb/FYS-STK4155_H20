import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import activation_functions as af
import common


def to_categorical(y, num_classes=None, dtype='float32'):
    """
    This function is copied from from tensorflow.keras.utils to reduce
    import time.

    Convert a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    Parameters
    ----------
    y : numpy.ndarray
        Class vector to be converted into a matrix (integers from 0 to
        num_classes).
    
    num_classes : NoneType, int 
        Total number of classes. If `None`, this would be inferred as
        the (largest number in `y`) + 1.
    
    dtype : str
        The data type expected by the input. Default: `'float32'`.

    Returns
    -------
    A binary matrix representation of the input. The classes axis is
    placed last.

    Raises
    ------
    ValueError: If input contains string value.
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class _FFNN:
    """
    Class implementation of a feedforward neural network.
    """
    def __init__(self,
        input_data,
        true_output,
        hidden_layer_sizes,
        n_categories,
        n_epochs,
        batch_size,
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.softmax,
        cost_function_derivative = af.cross_entropy_derivative_with_softmax,
        scaling = True,
        verbose = False,
        debug = False):
        """
        input_data : numpy.ndarray
            Matrix where each column is one set of dependent variables.

        true_output : numpy.ndarray
            The true output of the dependent variable(s).

        hidden_layer_sizes : tuple, list, numpy.ndarray
            List of number of nodes in each hidden layer.

        n_categories : int
            The number of output categories.  1 for regression.

        n_epochs : int
            The number of epochs.  One epoch is an iteration through
            'n_data_total' amount of data points.

        batch_size : int
            The amount of data in each minibatch.

        hidden_layer_activation_function : callable
            Activation function for the hidden layers.

        hidden_layer_activation_function_derivative : callable
            Derivative of the activation function for the hidden layers.

        output_activation_function : callable
            The output activation function.

        cost_function_derivative : callable
            Derivative of the cost function.

        scaling : bool
            Toggle scaling of input_data and true_output on / off.

        verbose : bool
            Toggle verbose mode on / off.

        debug : bool
            Toggle debug mode on / off.  Debug mode prints shape
            information and breaks the training loop after one
            iteration.
        """
        try:
            self.hidden_layer_sizes = hidden_layer_sizes
            self.n_hidden_layers = len(self.hidden_layer_sizes)
        except TypeError:
            msg = f"hidden_layer_sizes must be of type {tuple}, {list},"
            msg += f" or {np.ndarray}. Got {type(hidden_layer_sizes)}."
            print(msg)
            sys.exit()

        if not callable(hidden_layer_activation_function):
            msg = f"hidden_layer_activation_function must be callable!"
            msg += f" Got {type(hidden_layer_activation_function)}."
            print(msg)
            sys.exit()

        if not callable(hidden_layer_activation_function_derivative):
            msg = f"hidden_layer_activation_function must be callable!"
            msg += f" Got {type(hidden_layer_activation_function_derivative)}."
            print(msg)
            sys.exit()

        if not callable(output_activation_function):
            msg = f"output_activation_function must be callable!"
            msg += f" Got {type(output_activation_function)}."
            print(msg)
            sys.exit()

        if not callable(cost_function_derivative):
            msg = f"cost_function must be callable!"
            msg += f" Got {type(cost_function_derivative)}."
            print(msg)
            sys.exit()

        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.hidden_layer_activation_function_derivative = hidden_layer_activation_function_derivative
        self.output_activation_function = output_activation_function
        self.cost_function_derivative = cost_function_derivative

        self.X = input_data
        self.y = true_output
        self.n_data_total = self.X.shape[0] # Total number of data points.
        self.X = self.X.reshape(self.n_data_total, -1)
        self.n_features = self.X.shape[1]   # The number of features.
        
        self.n_epochs = n_epochs            # Number of epochs.
        self.batch_size = batch_size        # Size of each minibatch.
        self.n_categories = n_categories    # Number of output categories.
        self.scaling = scaling
        self.verbose = verbose
        self.debug = debug


    def _initial_state(self, subclass):
        """
        Set the system to the correct state before training starts.
        Split the data into training and testing sets.  Initialize the
        weights and biases for the hidden layer(s) and the output layer.

        Parameters
        ----------
        subclass : str
            _initial_state implementation in subclasses passes either
            'classifier' or 'regressor' to change shape of y_train and
            y_test.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, shuffle=True) 

        if self.scaling:
            self.X_mean = np.mean(self.X_train)
            self.X_std = np.std(self.X_train)
            self.X_train = (self.X_train - self.X_mean)/self.X_std
            self.X_test = (self.X_test - self.X_mean)/self.X_std

            self.y_mean = np.mean(self.y_train)
            self.y_std = np.std(self.y_train)
            self.y_train = (self.y_train - self.y_mean)/self.y_std
            self.y_test = (self.y_test - self.y_mean)/self.y_std
        
        if subclass == "classifier":
            """
            One-hot for classification problems.
            """
            self.y_train = to_categorical(self.y_train)
        
        elif subclass == "regressor":
            """
            If y_train is not a one-hot matrix, it needs to be a column
            vector.
            """
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)    # I spent ONE WEEK debugging, only to find that this single line of code solved literally everything.

        self.hidden_weights = np.zeros(shape=self.n_hidden_layers, dtype=np.ndarray)
        self.hidden_biases = np.zeros(shape=self.n_hidden_layers, dtype=np.ndarray)

        # Special case for the first hidden layer.
        self.hidden_weights[0] = np.random.normal(
            size=(self.n_features, self.hidden_layer_sizes[0]))
        self.hidden_biases[0] = np.full(
            shape=self.hidden_layer_sizes[0], fill_value=0.01)

        for i in range(1, self.n_hidden_layers):
            """
            Initialize weights and biases for all hidden layers except
            the first, which was handled before the loop as a special
            case.  The number of rows in the i'th layers hidden weights
            is equal to the number of neurons in the i-1'th layer.
            """
            self.hidden_weights[i] = np.random.normal(
                size=(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i]))
            self.hidden_biases[i] = np.full(
                shape=self.hidden_layer_sizes[i], fill_value=0.01)

        # Weights and biases for the output layer.
        self.output_weights = np.random.normal(
            size=(self.hidden_layer_sizes[-1], self.n_categories))
        self.output_biases = np.full(shape=self.n_categories, fill_value=0.01)


    def start_timing(self):
        self.stopwatch = time.time()


    def stop_timing(self):
        self.stopwatch = time.time() - self.stopwatch
        print(f"{sys._getframe().f_back.f_code.co_name} time: {self.stopwatch:.4f} s")


    def feedforward(self):
        """
        Perform one feedforward.
        """
        self.neuron_activation = np.zeros(shape=self.n_hidden_layers + 2, dtype=np.ndarray)  # a
        self.neuron_input = np.zeros(shape=self.n_hidden_layers + 2, dtype=np.ndarray) # z

        self.neuron_activation[0] = self.X_selection
        self.neuron_input[0] = np.array([0])

        for i in range(self.n_hidden_layers):
            """
            Loop over the hidden layers.  Calculate the neuron
            activation and neuron input for all neurons in all hidden
            layers.
            """
            self.neuron_input[i + 1] = self.neuron_activation[i]@self.hidden_weights[i] + self.hidden_biases[i]
            self.neuron_activation[i + 1] = self.hidden_layer_activation_function(self.neuron_input[i + 1])

        self.neuron_input[-1] = (self.neuron_activation[-2]@self.output_weights + self.output_biases)
        self.neuron_activation[-1] = self.output_activation_function(self.neuron_input[-1])


    def _backpropagation(self):
        """
        Perform one backpropagation using gradient descent.
        """
        self.error = np.zeros(shape=self.n_hidden_layers + 1, dtype=np.ndarray)
        self.error[-1] = self.cost_function_derivative(self.neuron_activation[-1], self.y_selection)
        self.error[-2] = self.output_weights@self.error[-1].T*\
            self.hidden_layer_activation_function_derivative(self.neuron_input[-2]).T

        self.bias_gradient = np.zeros(shape=self.n_hidden_layers + 1, dtype=np.ndarray)
        self.bias_gradient[-1] = np.sum(self.error[-1], axis=0)
        self.bias_gradient[-2] = np.sum(self.error[-2], axis=1)

        self.weight_gradient = np.zeros(shape=self.n_hidden_layers + 1, dtype=np.ndarray)
        self.weight_gradient[-1] = (self.error[-1].T@self.neuron_activation[-2]).T
        self.weight_gradient[-2] = (self.error[-2]@self.neuron_activation[-3])

        for i in range(-3, -self.n_hidden_layers - 2, -1):
            """
            Loop backwards through the errors, bias and weight
            gradients.
            """
            self.error[i] = self.hidden_weights[i + 2]@self.error[i + 1]*\
                self.hidden_layer_activation_function_derivative(self.neuron_input[i].T)
            self.bias_gradient[i] = np.sum(self.error[i], axis=1)
            self.weight_gradient[i] = self.error[i]@self.neuron_activation[i - 1]

        self.output_weights -= self.learning_rate*(self.weight_gradient[-1]) +\
            self.lambd*self.output_weights
        self.output_biases -= self.learning_rate*(self.bias_gradient[-1])

        for i in range(-1, -self.n_hidden_layers - 1, -1):
            """
            Loop backwards through the hidden weights and biases.
            """
            self.hidden_weights[i] -= self.learning_rate*(self.weight_gradient[i - 1].T) +\
                self.lambd*(self.hidden_weights[i])
            self.hidden_biases[i] -= self.learning_rate*(self.bias_gradient[i - 1])

        if self.debug:
            print("BACKPROPAGATION")
            print(f"{self.error[-1].shape=}")
            print(f"{self.neuron_activation[-1].shape=}")
            print(f"{self.y_selection.shape=}")
            

    def train_neural_network(self, learning_rate=0.1, lambd=0):
        """
        Train the neural network.  Send the training data to
        _backpropagation in minibatches.

        Parameters
        ----------
        learning_rate : float
            The learning rate, aka. eta.

        lambd : float
            Regularization parameter, aka lambda.
        """
        self._initial_state()
        if self.verbose: self.start_timing()
        self.learning_rate = learning_rate
        self.lambd = lambd

        data_indices = np.arange(self.X_train.shape[0])
        n_iterations = self.n_data_total//self.batch_size

        for _ in range(self.n_epochs):
            """
            Loop over epochs.  One epoch is an iteration through
            'n_data_total' amount of data points.
            """
            for _ in range(n_iterations):
                """
                Loop over iterations.  The number of iterations is the
                number of data points in each batch.  Draw a set of
                random indices for each iteration.  These random indices
                constitutes one minibatch.
                """
                minibatch_indices = np.random.choice(data_indices,
                    size=self.batch_size, replace=True)

                self.X_selection = self.X_train[minibatch_indices]
                self.y_selection = self.y_train[minibatch_indices]

                self.feedforward()
                self._backpropagation()

                if self.debug: break
            if self.debug: break

        if self.verbose: self.stop_timing()


    def predict(self, X):
        """
        Predict by performing one feedforward using the input data of
        this function.

        Parameters
        ----------
        X : numpy.ndarray
            Data to predict.

        Returns
        -------
        self.neuron_activation[-1] : numpy.ndarray
            Prediction.  Be mindful of the shape.
        """
        self.X_selection = X
        self.feedforward()
        return self.neuron_activation[-1]


class FFNNClassifier(_FFNN):
    def _initial_state(self):
        super(FFNNClassifier, self)._initial_state(subclass="classifier")

    
    def score(self, X, y):
        """
        Generate the classification score.

        Parameters
        ----------
        X : numpy.ndarray
            Data to predict.

        y : numpy.ndarray
            Output data to compare with.  Usually the true output of X.

        Returns
        -------
        score : float
            The classification score using
            sklearn.metrics.accuracy_score.  1 is best, 0 is worst.
        """
        self.predict(X)
        score = accuracy_score(np.argmax(self.neuron_input[-1], axis=1), y)
        return score
        

class FFNNRegressor(_FFNN):
    def _initial_state(self):
        """
        Apply _initial_state with 'regressor' parameter to shape the y
        data correctly.
        """
        super(FFNNRegressor, self)._initial_state(subclass="regressor")

    
    def score(self):
        """
        Calculate MSE and r score for test and train datasets.
        """
        y_train_prediction = self.predict(self.X_train)
        y_test_prediction = self.predict(self.X_test)        
        self.mse_train = common.mean_squared_error(self.y_train, y_train_prediction)
        self.mse_test = common.mean_squared_error(self.y_test, y_test_prediction)
        self.r_train = common.r_squared(self.y_train, y_train_prediction)
        self.r_test = common.r_squared(self.y_test, y_test_prediction)

        return self.mse_train, self.mse_test, self.r_train, self.r_test