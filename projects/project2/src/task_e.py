import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import ray
import neural_network as nn
import activation_functions as af


def compare_logistic_regression_and_neural_network_classification_learning_rates():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    
    n_repetitions = 1
    n_categories = 10
    n_epochs = 20
    batch_size = 20
    n_learning_rates = 30
    learning_rates = np.linspace(0.004, 0.006, n_learning_rates)
    
    nn_scores = np.zeros(n_learning_rates)  # Neural network.
    lr_scores = np.zeros(n_learning_rates)  # Logistic regression.

    nn_classifier = nn.FFNNClassifier(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50,),
        n_categories = n_categories,
        n_epochs = n_epochs,
        batch_size = batch_size,
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.softmax,
        cost_function_derivative = af.cross_entropy_derivative_with_softmax,
        scaling = True,
        verbose = True,
        debug = False)

    lr_classifier = nn.FFNNLogisticRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(),
        n_categories = n_categories,
        n_epochs = n_epochs,
        batch_size = batch_size,
        output_activation_function = af.softmax,
        cost_function_derivative = af.cross_entropy_derivative_with_softmax,
        scaling = True,
        verbose = True,
        debug = False)

    for rep in range(n_repetitions):
        for i in range(n_learning_rates):
            print(f"\nrepetition {rep+1} of {n_repetitions}")
            print(f"{i+1} of {n_learning_rates}, {learning_rates[i]=}")
            nn_classifier.train_neural_network(learning_rate=learning_rates[i])
            nn_scores[i] += nn_classifier.score(nn_classifier.X_test, nn_classifier.y_test)
            lr_classifier.train_neural_network(learning_rate=learning_rates[i])
            lr_scores[i] += lr_classifier.score(lr_classifier.X_test, lr_classifier.y_test)
        
    nn_scores /= n_repetitions
    lr_scores /= n_repetitions

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(learning_rates*1e3, nn_scores, label="Neural network", color="grey")
    ax.plot(learning_rates*1e3, lr_scores, label="Logistic regression", color="black", linestyle="dashed")
    ax.legend(fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xlabel(r"$\eta[10^{-3}]$", fontsize=15)
    ax.set_ylabel("Score", fontsize=15)
    ax.set_title(f"NN max: {np.max(nn_scores):.4f}, LR max: {np.max(lr_scores):.4f}", fontsize=15)
    plt.savefig(fname="../fig/task_e_nn_lr_score_eta.png", dpi=300)
    plt.show()


def compare_logistic_regression_and_neural_network_classification_batch_sizes():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    
    n_repetitions = 40
    n_categories = 10
    n_epochs = 20
    batch_sizes = np.arange(10, 100+1, 5)
    n_batch_sizes = len(batch_sizes)
    learning_rate = 4.8e-3
    
    nn_scores = np.zeros(n_batch_sizes)  # Neural network.
    lr_scores = np.zeros(n_batch_sizes)  # Logistic regression.

    nn_classifier = nn.FFNNClassifier(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(50,),
        n_categories = n_categories,
        n_epochs = n_epochs,
        batch_size = batch_sizes[0],
        hidden_layer_activation_function = af.sigmoid,
        hidden_layer_activation_function_derivative = af.sigmoid_derivative,
        output_activation_function = af.softmax,
        cost_function_derivative = af.cross_entropy_derivative_with_softmax,
        scaling = True,
        verbose = True,
        debug = False)

    lr_classifier = nn.FFNNLogisticRegressor(
        input_data = X,
        true_output = y,
        hidden_layer_sizes=(),
        n_categories = n_categories,
        n_epochs = n_epochs,
        batch_size = batch_sizes[0],
        output_activation_function = af.softmax,
        cost_function_derivative = af.cross_entropy_derivative_with_softmax,
        scaling = True,
        verbose = True,
        debug = False)

    for rep in range(n_repetitions):
        for i in range(n_batch_sizes):
            print(f"\nrepetition {rep+1} of {n_repetitions}")
            print(f"{i+1} of {n_batch_sizes}, {batch_sizes[i]=}")
            
            nn_classifier.batch_size = batch_sizes[i]
            nn_classifier.train_neural_network(learning_rate=learning_rate)
            nn_scores[i] += nn_classifier.score(nn_classifier.X_test, nn_classifier.y_test)
            
            lr_classifier.batch_size = batch_sizes[i]
            lr_classifier.train_neural_network(learning_rate=learning_rate)
            lr_scores[i] += lr_classifier.score(lr_classifier.X_test, lr_classifier.y_test)
        
    nn_scores /= n_repetitions
    lr_scores /= n_repetitions

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(batch_sizes, nn_scores, label="Neural network", color="grey")
    ax.plot(batch_sizes, lr_scores, label="Logistic regression", color="black", linestyle="dashed")
    ax.legend(fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xlabel(r"Batch size", fontsize=15)
    ax.set_ylabel("Score", fontsize=15)
    ax.set_title(f"NN max: {np.max(nn_scores):.4f}, LR max: {np.max(lr_scores):.4f}", fontsize=15)
    plt.savefig(fname="../fig/task_e_nn_lr_score_batch_size.png", dpi=300)
    plt.show()


def compare_logistic_regression_and_neural_network_classification_n_epochs():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    
    n_repetitions = 15
    n_categories = 10
    epoch_range = [10, 400]
    n_epochs = np.arange(epoch_range[0], epoch_range[1]+1, 5)
    n_n_epochs = len(n_epochs)
    batch_size = 20
    learning_rate = 4.8e-3
    hidden_layer_sizes = (50,)  # Only for NN.
    name = f"_{n_repetitions=}_{n_n_epochs=}_epoch_range=[{epoch_range[0]}_{epoch_range[1]}]_{learning_rate=}_{batch_size=}_{hidden_layer_sizes=}.npy"
    # name = f"{n_repetitions=}_{n_n_epochs=}_{learning_rate=}.npy"

    try:
        nn_scores = np.load(file="data_files/task_e_nn_scores" + name)
        lr_scores = np.load(file="data_files/task_e_lr_scores" + name)
        nn_times = np.load(file="data_files/task_e_nn_times" + name)
        lr_times = np.load(file="data_files/task_e_lr_times" + name)
    
    except FileNotFoundError:
        nn_scores = np.zeros(n_n_epochs)  # Neural network.
        lr_scores = np.zeros(n_n_epochs)  # Logistic regression.
        nn_times = np.zeros(n_n_epochs)
        lr_times = np.zeros(n_n_epochs)

        ray.init()  
        @ray.remote
        def epoch_func():
            nn_classifier = nn.FFNNClassifier(
                input_data = X,
                true_output = y,
                hidden_layer_sizes=hidden_layer_sizes,
                n_categories = n_categories,
                n_epochs = n_epochs,
                batch_size = batch_size,
                hidden_layer_activation_function = af.sigmoid,
                hidden_layer_activation_function_derivative = af.sigmoid_derivative,
                output_activation_function = af.softmax,
                cost_function_derivative = af.cross_entropy_derivative_with_softmax,
                scaling = True,
                verbose = True,
                debug = False)

            lr_classifier = nn.FFNNLogisticRegressor(
                input_data = X,
                true_output = y,
                hidden_layer_sizes = (),
                n_categories = n_categories,
                n_epochs = n_epochs,
                batch_size = batch_size,
                output_activation_function = af.softmax,
                cost_function_derivative = af.cross_entropy_derivative_with_softmax,
                scaling = True,
                verbose = True,
                debug = False)

            nn_scores_tmp = np.zeros(n_n_epochs)  # Neural network.
            lr_scores_tmp = np.zeros(n_n_epochs)  # Logistic regression.
            nn_times_tmp = np.zeros(n_n_epochs)
            lr_times_tmp = np.zeros(n_n_epochs)
            for i in range(n_n_epochs):
                print(f"\nrepetition {rep+1} of {n_repetitions}")
                print(f"{i+1} of {n_n_epochs}, {n_epochs[i]=}")
                
                nn_classifier.n_epochs = n_epochs[i]
                nn_classifier.train_neural_network(learning_rate=learning_rate)
                nn_scores_tmp[i] = nn_classifier.score(nn_classifier.X_test, nn_classifier.y_test)
                nn_times_tmp[i] = nn_classifier.stopwatch
                
                lr_classifier.n_epochs = n_epochs[i]
                lr_classifier.train_neural_network(learning_rate=learning_rate)
                lr_scores_tmp[i] = lr_classifier.score(lr_classifier.X_test, lr_classifier.y_test)
                lr_times_tmp[i] = nn_classifier.stopwatch

            return nn_scores_tmp, lr_scores_tmp, nn_times_tmp, lr_times_tmp
        
        parallels = []
        for rep in range(n_repetitions):
            parallels.append(epoch_func.remote())
            # for i in range(n_n_epochs):
            #     print(f"\nrepetition {rep+1} of {n_repetitions}")
            #     print(f"{i+1} of {n_n_epochs}, {n_epochs[i]=}")
                
            #     nn_classifier.n_epochs = n_epochs[i]
            #     nn_classifier.train_neural_network(learning_rate=learning_rate)
            #     nn_scores[i] += nn_classifier.score(nn_classifier.X_test, nn_classifier.y_test)
            #     nn_times[i] += nn_classifier.stopwatch
                
            #     lr_classifier.n_epochs = n_epochs[i]
            #     lr_classifier.train_neural_network(learning_rate=learning_rate)
            #     lr_scores[i] += lr_classifier.score(lr_classifier.X_test, lr_classifier.y_test)
            #     lr_times[i] += nn_classifier.stopwatch

        for res in ray.get(parallels):
            nn_scores_tmp, lr_scores_tmp, nn_times_tmp, lr_times_tmp = res
            nn_scores += nn_scores_tmp
            lr_scores += lr_scores_tmp
            nn_times += nn_times_tmp
            lr_times += lr_times_tmp
            
        nn_scores /= n_repetitions
        lr_scores /= n_repetitions
        nn_times /= n_repetitions
        lr_times /= n_repetitions

        np.save(file="data_files/task_e_nn_scores" + name, arr=nn_scores)
        np.save(file="data_files/task_e_lr_scores" + name, arr=lr_scores)
        np.save(file="data_files/task_e_nn_times" + name, arr=nn_times)
        np.save(file="data_files/task_e_lr_times" + name, arr=lr_times)

    fig0, ax0 = plt.subplots(figsize=(9, 7))
    ax0.plot(n_epochs, nn_scores, label="Neural network", color="grey")
    ax0.plot(n_epochs, lr_scores, label="Logistic regression", color="black", linestyle="dashed")
    ax0.legend(fontsize=15)
    ax0.tick_params(labelsize=15)
    ax0.set_xlabel(r"# epochs", fontsize=15)
    ax0.set_ylabel("Score", fontsize=15)
    ax0.set_title(f"NN max: {np.max(nn_scores):.4f}, LR max: {np.max(lr_scores):.4f}", fontsize=15)
    ax0.grid()
    fig0.savefig(fname="../fig/task_e_nn_lr_score_n_epochs.png" + name[:-4] + ".png", dpi=300)
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.plot(nn_times[1:], nn_scores[1:], label="Neural network", color="grey")
    ax1.plot(lr_times[1:], lr_scores[1:], label="Logistic regression", color="black", linestyle="dashed")
    ax1.legend(fontsize=15)
    ax1.tick_params(labelsize=15)
    ax1.set_xlabel(r"Time(# batches) [s]", fontsize=15)
    ax1.set_ylabel("Score", fontsize=15)
    ax1.set_title(f"NN max: {np.max(nn_scores):.4f}, LR max: {np.max(lr_scores):.4f}", fontsize=15)
    ax1.grid()
    fig1.savefig(fname="../fig/task_e_nn_lr_score_time" + name[:-4] + ".png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # compare_logistic_regression_and_neural_network_classification_learning_rates()
    # compare_logistic_regression_and_neural_network_classification_batch_sizes()
    compare_logistic_regression_and_neural_network_classification_n_epochs()
    pass