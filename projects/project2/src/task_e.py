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
        prediction = np.argmax(af.softmax(q1.X_test@q1.output_weights), axis=1)

        print(accuracy_score(prediction, q1.y_test))
    #     scores[i] = q1.score(q1.X_test, q1.y_test)
    #     print(f"{learning_rates[i]=}")
    #     print(f"{scores[i]=}\n")

    # plt.plot(learning_rates, scores)
    # plt.xlabel("learning rate")
    # plt.ylabel("score")
    # plt.show()


def logistic_regression_classification_2():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    n_data_total = X.shape[0]
    X = X.reshape(n_data_total, -1)
    n_features = X.shape[1]
    n_epochs = 100
    batch_size = 20
    n_learning_rates = 20
    learning_rates = np.linspace(1e-4, 1e-1, n_learning_rates)
    n_iterations = n_data_total//batch_size
    n_categories = 10

    print(f"{n_iterations=}")


    for i in range(n_learning_rates):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, shuffle=True)

        X_mean = np.mean(X_train)
        X_std = np.std(X_train)
        X_train = (X_train - X_mean)/X_std
        X_test = (X_test - X_mean)/X_std

        y_train = nn.to_categorical(y_train)

        output_weights = np.random.normal(size=(n_features, n_categories))
        neuron_activation = np.zeros(shape=2, dtype=np.ndarray)  # a
        neuron_input = np.zeros(shape=2, dtype=np.ndarray) # z
        
        data_indices = np.arange(X_train.shape[0])
        for _ in range(n_epochs):
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
                    size=batch_size, replace=True)

                X_selection = X_train[minibatch_indices]
                y_selection = y_train[minibatch_indices]

                # FEEDFORWARD
                neuron_activation[0] = X_selection
                neuron_input[0] = np.array([0])
                neuron_input[-1] = X_selection@output_weights
                neuron_activation[-1] = af.softmax(neuron_input[-1])
                # FEEDFORWARD
                
                # error = y_selection - neuron_activation[-1]
                error = neuron_activation[-1] - y_selection
                weight_gradient = neuron_activation[-2].T@error
                # weight_gradient = -(error.T@neuron_activation[-2]).T
                output_weights -= learning_rates[i]*weight_gradient

        prediction = np.argmax(af.softmax(X_test@output_weights), axis=1)

        print(accuracy_score(prediction, y_test))


if __name__ == "__main__":
    # neural_network_classification()
    logistic_regression_classification()
    # logistic_regression_classification_2()
    pass


                # exponential = np.exp(X_selection @ beta)
                # exponential = exponential / (1 + exponential)
                # gradient = (-X_selection.T@(y_selection - exponential)).sum(axis=1).reshape(-1, 1)
                # beta -= learning_rates[i]*gradient*2/(batch_size*y_train.shape[1])