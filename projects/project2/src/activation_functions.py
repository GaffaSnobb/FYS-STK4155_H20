import numba
import numpy as np

@numba.njit
def sigmoid(x):
    """
    Sigmoid function.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input parameter.
    """
    return 1/(1 + np.exp(-x))


@numba.njit
def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function.

    Parameters
    ----------
    x : numpy.ndarray
        Input parameter.
    """
    val = sigmoid(x)
    return val*(1 - val)


@numba.njit
def cross_entropy_derivative_with_softmax(y_predicted, y_actual):
    """
    Derivative of cross entropy cost function.
    """
    return y_predicted - y_actual


# @numba.njit # Tests fail with jit.
def softmax(x):
    """
    Softmax activation function.
    z : numpy.ndarray
        Input variable.
    """
    exponential_term = np.exp(x)
    return exponential_term/np.sum(exponential_term, axis=1, keepdims=True)


@numba.njit
def linear(x):
    return x


def relu(x): 
    return np.maximum(0, x)


def relu_derivative(x):
    return (0 < x).astype(int)


def leaky_relu(x):
    y = np.copy(x)  # I think we must do this to not alter the actual x values.
    y[y < 0] *= 0.01
    return y


def leaky_relu_derivative(x):
    y = np.copy(x)  # I think we must do this to not alter the actual x values.
    y[y > 0] = 1
    y[y < 0] = 0.01
    return y


@numba.njit
def mse_derivative(y_predicted, y_actual):
    return 2/y_actual.shape[0]*(y_predicted - y_actual)