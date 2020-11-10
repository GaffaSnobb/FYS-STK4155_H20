import numpy as np
from sklearn import datasets
import common

def classification():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    
    q1 = common.FFNNClassifier(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50, 20, 20),
        n_categories = 10,
        n_epochs = 20,
        batch_size = 20,
        hidden_layer_activation_function = common.sigmoid,
        output_activation_function = common.softmax,
        cost_function = common.cross_entropy_derivative,
        verbose = True,
        debug = False)
    
    q1.train_neural_network(learning_rate=0.007)
    score = q1.predict(q1.X_test)
    print(score)


# def regression():
#     n_data_total = 400
#     x1 = np.random.uniform(0, 1, n_data_total)
#     x2 = np.random.uniform(0, 1, n_data_total)
#     X = np.zeros(shape=(n_data_total, 2))
#     for i in range(n_data_total): X[i] = x1[i], x2[i]
#     y = common.franke_function(x1, x2)

    
#     q1 = common.FFNN(
#         input_data = X,
#         true_output = y,
#         hidden_layer_sizes=(50,),
#         n_categories = 1,
#         n_epochs = 20,
#         batch_size = 20,
#         hidden_layer_activation_function = common.sigmoid,
#         output_activation_function = common.softmax,
#         cost_function = common.cross_entropy_derivative,
#         verbose = True,
#         debug = False)

#     q1.train_neural_network()


if __name__ == "__main__":
    classification()
    # regression()

    # for learning_rate in np.logspace(-5, 0, 8):
    #     q1.train_neural_network_single(learning_rate)
    #     score = q1.predict_single(q1.X_test)
    #     print(f"score: {score} for learning rate: {learning_rate}")
    pass