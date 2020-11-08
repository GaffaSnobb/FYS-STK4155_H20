import numpy as np
from sklearn import datasets
import common

# class FFNNRegression(common.FFNN):
#     def __init__(self, n_data_total, poly_degree, init_beta=None):
#         """
#         Solve a 2D example of Franke data using a feedforward neural
#         network.

#         Parameters
#         ----------
#         n_data_total : int
#             The number of data points.

#         poly_degree : int
#             The polynomial degree.
#         """
#         self.n_dependent_variables = 2
#         self.x1 = np.random.uniform(0, 1, n_data_total)
#         self.x2 = np.random.uniform(0, 1, n_data_total)

#         self.X = common.create_design_matrix_two_dependent_variables(self.x1,
#             self.x2, n_data_total, poly_degree)
#         self.y = common.franke_function(self.x1, self.x2)
        
#         # super(FFNNRegression, self).__init__(n_data_total, poly_degree, init_beta)


def neural_network_franke():
    digits = datasets.load_digits()
    # X = digits.images
    # y = digits.target
    n_data_total = 200
    poly_degree = 3

    x1 = np.random.uniform(0, 1, n_data_total)
    x2 = np.random.uniform(0, 1, n_data_total)
    X = common.create_design_matrix_two_dependent_variables(x1,
        x2, n_data_total, poly_degree)
    y = common.franke_function(x1, x2)
    n_features = X.shape[1]

    # print(X.shape)
    # print(y.shape)
    # print(X.reshape(X.shape[0], -1).shape)
    # print(digits.images.shape)
    # print(digits.target.shape)
    # print(digits.images.reshape(digits.images.shape[0], -1).shape)
    
    q1 = common.FFNN(X=X, y=y,
        hidden_layer_sizes=(50, 20, 20), n_categories=1,
        hidden_layer_activation_function=common.sigmoid,
        output_activation_function=common.linear,
        cost_function=common.cross_entropy_derivative,
        verbose=False)
    
    # q1 = common.FFNN(X=digits.images, y=digits.target,
    #     hidden_layer_sizes=(50, 20, 20), n_categories=10,
    #     hidden_layer_activation_function=common.sigmoid,
    #     output_activation_function=common.softmax,
    #     cost_function=common.cross_entropy_derivative,
    #     verbose=True)
    
    q1.train_neural_network(learning_rate=0.007)
    score = q1.predict_regression(q1.X_test)
    for predicted, actual in zip(score, q1.y_test):
        print(predicted, actual)
    print(score.shape)
    print(q1.y_test.shape)


if __name__ == "__main__":
    # digits = datasets.load_digits()
    # X = digits.images
    # y = digits.target
    
    # q1 = common.FFNN(X=X, y=y, hidden_layer_sizes=(50, 20, 20),
    #     hidden_layer_activation_function=common.sigmoid, verbose=True)
    
    # q1.train_neural_network(learning_rate=0.007)
    # score = q1.predict(q1.X_test)
    # print(score)

    neural_network_franke()

    # for learning_rate in np.logspace(-5, 0, 8):
    #     q1.train_neural_network_single(learning_rate)
    #     score = q1.predict_single(q1.X_test)
    #     print(f"score: {score} for learning rate: {learning_rate}")
    pass