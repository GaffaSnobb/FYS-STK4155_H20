import numpy as np
from sklearn import datasets
import common

def regression():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    
    q1 = common.FFNN(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50, 20, 20),
        n_categories = 10,
        n_epochs = 20,
        batch_size = 20,
        hidden_layer_activation_function=common.sigmoid,
        output_activation_function=common.linear,
        cost_function = common.cross_entropy_derivative,
        verbose=True,
        debug = False)
    
    q1.train_neural_network(learning_rate=0.007)
    score = q1.predict(q1.X_test)
    print(score)

if __name__ == "__main__":
    regression()

    # for learning_rate in np.logspace(-5, 0, 8):
    #     q1.train_neural_network_single(learning_rate)
    #     score = q1.predict_single(q1.X_test)
    #     print(f"score: {score} for learning rate: {learning_rate}")
    pass