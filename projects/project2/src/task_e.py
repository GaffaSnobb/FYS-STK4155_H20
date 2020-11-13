import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import neural_network as nn
import activation_functions as af
import common

def neural_network_classification():
    """
    Only used for testing the neural network, not directly related to
    task_b.
    """
    np.random.seed(1337)
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    
    q1 = nn.FFNNClassifier(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50,),
        n_categories = 10,
        n_epochs = 50,
        batch_size = 20,
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.softmax,
        cost_function_derivative = af.cross_entropy_derivative_with_softmax,
        scaling = False,
        verbose = True,
        debug = False)

    n_learning_rates = 10
    learning_rates = np.linspace(1e-3, 1e-2, n_learning_rates)
    scores = np.zeros(n_learning_rates)

    for i in range(n_learning_rates):
        q1.train_neural_network(learning_rate=learning_rates[i])
        scores[i] = q1.score(q1.X_test, q1.y_test)
        print(scores[i])

    plt.plot(learning_rates, scores)
    plt.xlabel("learning rate")
    plt.ylabel("score")
    plt.show()


def logistic_regression_classification():
    # np.random.seed(1337)
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target

    q1 = nn.FFNNLogisticRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(),  # No hidden layers for logistic!
        n_categories = 10,
        n_epochs = 100,
        batch_size = 20,
        output_activation_function = af.softmax,
        cost_function_derivative = af.cross_entropy_derivative_with_softmax,
        verbose = True,
        debug = False,
        scaling = True)

    n_learning_rates = 20
    learning_rates = np.linspace(1e-4, 1e-1, n_learning_rates)
    scores = np.zeros(n_learning_rates)

    for i in range(n_learning_rates):
        q1.train_neural_network(learning_rate=learning_rates[i], lambd=0)
        # prediction = np.argmax(af.softmax(q1.X_test@q1.output_weights), axis=1)

        # print(accuracy_score(prediction, q1.y_test))
        scores[i] = q1.score(q1.X_test, q1.y_test)
        print(f"{learning_rates[i]=}")
        print(f"{scores[i]=}\n")

    plt.plot(learning_rates, scores)
    plt.xlabel("learning rate")
    plt.ylabel("score")
    plt.show()


if __name__ == "__main__":
    # neural_network_classification()
    logistic_regression_classification()
    pass