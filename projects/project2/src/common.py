import time
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def polynomial_1d(x, *beta):
    """
    n'th degree polynomial for fit testing.

    Parameters
    ----------
    x : numpy.ndarray
        Dependent variable.

    *beta : floats
        Polynomial coefficients.
    """
    res = np.zeros(len(x))
    for i in range(len(beta)):
        res += beta[i]*x**i

    return res


def mean_squared_error(x, y):
    """
    Calculate the mean squared error.

    Consider adding the length n as an argument if this function is
    called many times.

    Parameters
    ----------
    x : numpy.ndarray
        Value 1.

    y : numpy.ndarray
        Value 2.

    Returns
    -------
    : numpy.ndarray
        The mean squared error.
    """
    return np.mean((x - y)**2)


def r_squared(y_observed, y_predicted):
    """
    Calculate the score R**2.
    
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
    return 1 - np.sum((y_observed - y_predicted)**2)/\
        np.sum((y_observed - np.mean(y_observed))**2)


def step_length(t, t0, t1):
    return t0/(t + t1)


def franke_function(x, y):
    return 0.75*np.exp(-(9*x - 2)**2/4 - (9*y - 2)**2/4) + \
        0.75*np.exp(-(9*x + 1)**2/49 - (9*y + 1)/10) + \
        0.5*np.exp(-(9*x - 7)**2/4 - (9*y - 3)**2/4) - \
        0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)


def features(degree, n_dependent_variables):
    """
    Calculate the number of features for a given polynomial degree for
    one or two dependent variables.

    Parameters
    ----------
    degree : int
        Polynomial degree.

    n_dependent_variables : int
        The number of dependent variables.
    """
    allowed = [1, 2]
    success = n_dependent_variables in allowed
    msg = "The number of dependent variables must be one of the following:"
    msg += f" {allowed}"
    assert success, msg
    
    if n_dependent_variables == 1:
        return degree + 1
    elif n_dependent_variables == 2:
        return int((degree + 1)*(degree + 2)/2)


def create_design_matrix_two_dependent_variables(x1, x2, N, degree):
    """
    Construct a design matrix with N rows and features =
    (degree + 1)*(degree + 2)/2 columns.  N is the number of samples and
    features is the number of features of the design matrix.  For two
    dependent variables.

    Parameters
    ----------

    x1 : numpy.ndarray
        Dependent variable.

    x2 : numpy.ndarray
        Dependent variable.

    N : int
        The number of data ponts.

    degree : int
        The polynomial degree.

    Returns
    -------
    X : numpy.ndarray
        Design matrix of dimensions N rows and (degree + 1)*(degree + 2)/2
        columns.
    """
    
    X = np.empty((N, features(degree, 2)))     # Data points x features.
    X[:, 0] = 1 # Intercept.
    col_idx = 1 # For indexing the design matrix columns.

    for j in range(1, degree+1):
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


def create_design_matrix_one_dependent_variable(x, n_data_total, poly_degree):
    """
    Construct a design matrix with 'n_data_total' rows and
    'poly_degree+1' columns.  For one dependent variable.

    Parameters
    ----------
    x : numpy.ndarray
        Dependent variable.

    n_data_total : int
        The number of data points (rows).

    poly_degree : int
        The polynomial degree (cols-1).

    Returns
    -------
    X : numpy.ndarray
        Design matrix.
    """
    X = np.empty((n_data_total, features(poly_degree, 1)))
    X[:, 0] = 1 # Intercept.

    for i in range(1, poly_degree+1):
        X[:, i] = x**i

    return X


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


def sigmoid(x):
    """
    Sigmoid function.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input parameter.
    """
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function.

    Parameters
    ----------
    x : numpy.ndarray
        Input parameter.
    """
    # exponential_term = np.exp(-x)
    # return exponential_term/(1 + exponential_term)**2
    val = sigmoid(x)
    return val*(1 - val)


def cross_entropy_derivative_with_softmax(y_predicted, y_actual):
    """
    Derivative of cross entropy cost function.
    """
    return y_predicted - y_actual


def softmax(z):
    """
    Softmax activation function.
    z : numpy.ndarray
        Input variable.
    """
    exponential_term = np.exp(z)
    return exponential_term/np.sum(exponential_term, axis=1, keepdims=True)


def linear(z):
    return z


def relu(x): 
    return np.maximum(0, x)


def relu_derivative(x):
    return (0 < x).astype("int")


def mse_derivative(y_predicted, y_actual):
    return 2/y_actual.shape[0]*(y_predicted - y_actual)


class _StatTools:
    def __init__(self, n_data_total, poly_degree, init_beta=None):
        """
        Parameters
        ----------
        n_data_total : int
            The total number of data points.

        poly_degree : int
            The polynomial degree.

        init_beta : NoneType, numpy.ndarray
            Initial beta values.  Defaults to None where 0 is used.
        """
        self.n_data_total = n_data_total
        self.poly_degree = poly_degree
        self.n_features = features(self.poly_degree, self.n_dependent_variables)
        self.init_beta = init_beta

        self._split_scale()


    def gradient_descent(self, iterations, step_size):
        """
        Solve for beta using gradient descent.

        Parameters
        ----------
        iterations : int
            The number of iterations of the gradient descent.
        
        step_size : int
            The step size of the gradient descent.  AKA learning rate.
        """
        self.reset_state()    # Reset beta for every new GD.
        if self.verbose: self.start_timing()
        
        for _ in range(iterations):
            """
            Loop over the gradient descents.
            """
            gradient = self.X_train.T@(self.X_train@self.beta - self.y_train)
            gradient *= 2/self.n_data_total
            self.beta -= step_size*gradient

        if self.verbose: self.stop_timing()


    def stochastic_gradient_descent(self, n_epochs, n_batches,
        input_step_size=None, lambd=0):
        """
        Solve for beta using stochastic gradient descent with momentum.

        Parameters
        ----------
        n_epochs : int
            The number of epochs.

        n_batches : int
            The number of batches.  If the number of rows in the design
            matrix does not divide by n_batches, the rest rows are
            discarded.

        lambd : float
            Ridge regression penalty parameter.  Defaults to 0 where no
            ridge penalty is applied.

        input_step_size : NoneType, float
            The gradient step size / learning rate.  Defaults to None
            where a dynamic step size is used.
        """
        self.reset_state()    # Reset beta for every new SGD.
        if self.verbose: self.start_timing()
        
        rest = self.n_data_total%n_batches # The rest after equally splitting X into batches.
        n_data_per_batch = self.n_data_total//n_batches # Index step size.
        # Indices of X corresponding to start point of the batches.
        batch_indices = np.arange(0, self.n_data_total-rest, n_data_per_batch)

        momentum_parameter = 0.5
        momentum = 0

        for epoch in range(n_epochs):
            """
            Loop over epochs.
            """
            t_step = epoch*n_data_per_batch   # Does not need to be calculated in the inner loop.
            
            # for i in range(n_data_per_batch):
            for i in range(n_batches):
                """
                Loop over all data in each batch.  For each loop, a
                random start index defined by the number of batches,
                is drawn.  This chooses a random batch by slicing the
                design matrix.
                """
                random_index = np.random.choice(batch_indices)
                X = self.X_train[random_index:random_index+n_data_per_batch]
                y = self.y_train[random_index:random_index+n_data_per_batch]
                t_step += i
                
                if input_step_size is None:
                    step_size = step_length(t=t_step, t0=5, t1=50)
                else: step_size = input_step_size

                gradient = X.T@((X@self.beta) - y)*2/n_data_per_batch
                gradient += 2*lambd*self.beta   # Ridge addition.
                momentum = momentum_parameter*momentum + step_size*gradient
                self.beta -= momentum

        if self.verbose: self.stop_timing()


    def _split_scale(self):
        """
        Split the data into training and test sets.  Scale the data by
        subtracting the mean and dividing by the standard deviation,
        both values from the training set.  Shuffle the values.
        """
        # Splitting.
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, shuffle=True)

        # Scaling.
        # self.X_mean = np.mean(self.X_train)
        # self.X_std = np.std(self.X_train)
        # self.X_train = (self.X_train - self.X_mean)/self.X_std
        # self.X_test = (self.X_test - self.X_mean)/self.X_std


    def reset_state(self):
        """
        Reset beta to the initial guess state and shuffle training data.
        """
        if self.init_beta is None:
            self.beta = np.zeros(self.n_features)
        else:
            msg = "Initial beta value array must be of length"
            msg += f" {self.n_features}, got {len(self.init_beta)}."
            success = len(self.init_beta) == (self.n_features)
            assert success, msg
            
            self.beta = self.init_beta

        state = np.random.get_state()
        np.random.shuffle(self.X_train)
        np.random.set_state(state)
        np.random.shuffle(self.y_train)


    def start_timing(self):
        self.stopwatch = time.time()


    def stop_timing(self):
        self.stopwatch = time.time() - self.stopwatch
        print(f"{sys._getframe().f_back.f_code.co_name} time: {self.stopwatch:.4f} s")

    @property
    def mse(self):
        mse_train = mean_squared_error(self.y_train, self.X_train@self.beta)
        mse_test = mean_squared_error(self.y_test, self.X_test@self.beta)

        return mse_train, mse_test


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
        hidden_layer_activation_function = sigmoid,
        hidden_layer_activation_function_derivative = sigmoid_derivative,
        output_activation_function = softmax,
        cost_function_derivative = cross_entropy_derivative_with_softmax,
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
        self.mse_train = mean_squared_error(self.y_train, y_train_prediction)
        self.mse_test = mean_squared_error(self.y_test, y_test_prediction)
        self.r_train = r_squared(self.y_train, y_train_prediction)
        self.r_test = r_squared(self.y_test, y_test_prediction)

        return self.mse_train, self.mse_test, self.r_train, self.r_test