import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import activation_functions as af
import neural_network as nn
import ray


def eta_lambda_score_heatmap():
    """
    Solve classification problem. Loop over regularization parameters
    and learning rates.
    """
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target

    n_repetitions = 15
    n_learning_rates = 5
    n_regularization_parameters = 5
    learning_rates = np.linspace(0.01, 0.06, n_learning_rates)   # eta
    regularization_parameters = np.linspace(1e-5, 1e-4, n_regularization_parameters)
    fname = f"data_files/task_d_lambda_eta_score.npy"

    try:
        scores = np.load(file=fname)
    except FileNotFoundError:

        scores = np.zeros(shape=(n_regularization_parameters, n_learning_rates))

        classifier = nn.FFNNClassifier(
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

        ray.init()
        @ray.remote
        def func():
            scores_tmp = np.zeros(shape=(n_regularization_parameters, n_learning_rates))
            for i in range(n_regularization_parameters):
                for j in range(n_learning_rates):
                    print(f"\n repetition {rep + 1:3d} of {n_repetitions}")
                    print(f"reg {i + 1} of {n_regularization_parameters}")
                    print(f"learn {j + 1} of {n_learning_rates}")
                    
                    classifier.train_neural_network(
                        learning_rate = learning_rates[j],
                        lambd = regularization_parameters[i])

                    scores_tmp[i, j] += classifier.score(classifier.X_test, classifier.y_test)

            return scores_tmp

        parallel = []
        for rep in range(n_repetitions):
            """
            The different processes are created here.
            """
            parallel.append(func.remote())

        for res in ray.get(parallel):
            """
            The parallel work is performed and extracted here.
            """
            scores += res
        
        scores /= n_repetitions
        
        np.save(file=fname, arr=scores)

    
    fig, ax = plt.subplots(figsize=(9, 7))
    ax = sns.heatmap(
        data = scores,
        linewidth = 0.5,
        annot = True,
        cmap = 'viridis',
        xticklabels = [f"{x:.2f}" for x in learning_rates],
        yticklabels = [f"{x*1e5:.2f}" for x in regularization_parameters],
        ax = ax,
        annot_kws = {"size": 14})

    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(labelsize=15)
    ax.set_xlabel(r"$\eta$", fontsize=15)
    ax.set_ylabel(r"$\lambda [10^{-5}]$", fontsize=15)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.set_ylabel('Accuracy', fontsize=15, rotation=90)
    fig.savefig(dpi=300, fname="../fig/task_d_lambda_eta_heatmap.png")
    plt.show()


def nodes_layers_score_heatmap():
    """
    Solve classification problem. Loop over the number of nodes per
    layer and the number of hidden layers.
    """
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target

    n_repetitions = 15
    n_nodes = np.arange(10, 200+1, 40)
    n_n_nodes = len(n_nodes)
    hidden_layers = np.arange(1, 3+1, 1)
    n_hidden_layers = len(hidden_layers)
    fname = f"data_files/task_d_nodes_layers_score.npy"

    try:
        scores = np.load(file=fname)
    except FileNotFoundError:

        scores = np.zeros(shape=(n_hidden_layers, n_n_nodes))

        ray.init()
        @ray.remote
        def func():
            scores_tmp = np.zeros(shape=(n_hidden_layers, n_n_nodes))
            for i in range(n_hidden_layers):
                for j in range(n_n_nodes):
                    print(f"\n repetition {rep + 1:3d} of {n_repetitions}")
                    print(f"hidden {i + 1} of {n_hidden_layers}")
                    print(f"nodes {j + 1} of {n_n_nodes}")

                    layers = [n_nodes[i]]*hidden_layers[i]

                    classifier = nn.FFNNClassifier(
                        input_data = X,
                        true_output = y,
                        hidden_layer_sizes = layers,
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
                    
                    classifier.train_neural_network(
                        learning_rate = 0.02,
                        lambd = 0)

                    scores_tmp[i, j] += classifier.score(classifier.X_test, classifier.y_test)

            return scores_tmp

        parallel = []
        for rep in range(n_repetitions):
            """
            The different processes are created here.
            """
            parallel.append(func.remote())

        for res in ray.get(parallel):
            """
            The parallel work is performed and extracted here.
            """
            scores += res
        
        scores /= n_repetitions
        
        np.save(file=fname, arr=scores)

    
    fig, ax = plt.subplots(figsize=(9, 7))
    ax = sns.heatmap(
        data = scores,
        linewidth = 0.5,
        annot = True,
        cmap = 'viridis',
        xticklabels = n_nodes,
        yticklabels = hidden_layers,
        ax = ax,
        annot_kws = {"size": 14})

    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(labelsize=15)
    ax.set_xlabel(r"#  of nodes per layer", fontsize=15)
    ax.set_ylabel(r"# hidden layers", fontsize=15)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.set_ylabel('Accuracy', fontsize=15, rotation=90)
    fig.savefig(dpi=300, fname="../fig/task_d_nodes_layers_heatmap.png")
    plt.show()


def nodes_activations_score_heatmap():
    """
    Solve classification problem. Loop over the number of nodes per
    layer and the number of hidden layers.
    """
    np.random.seed(1337)
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target

    n_repetitions = 15
    activation_functions = [af.sigmoid, np.tanh, af.relu]
    names = ["Logistic", "tanh", "ReLU"]
    learning_rates = [0.02, 0.001, 0.001]
    activation_functions_derivatives = [af.sigmoid_derivative, lambda x: (np.cosh(x))**(-2), af.relu_derivative]
    n_activation_functions = len(activation_functions)
    hidden_layers = np.arange(1, 3+1, 1)
    n_hidden_layers = len(hidden_layers)
    fname = f"data_files/task_d_activations_layers_score.npy"

    try:
        scores = np.load(file=fname)
    except FileNotFoundError:

        scores = np.zeros(shape=(n_hidden_layers, n_activation_functions))

        ray.init()
        @ray.remote
        def func():
            scores_tmp = np.zeros(shape=(n_hidden_layers, n_activation_functions))
            for i in range(n_hidden_layers):
                for j in range(n_activation_functions):
                    print(f"\n repetition {rep + 1:3d} of {n_repetitions}")
                    print(f"hidden {i + 1} of {n_hidden_layers}")
                    print(f"activations {j + 1} of {n_activation_functions}")

                    layers = [50]*hidden_layers[i]

                    classifier = nn.FFNNClassifier(
                        input_data = X,
                        true_output = y,
                        hidden_layer_sizes = layers,
                        n_categories = 10,
                        n_epochs = 50,
                        batch_size = 20,
                        hidden_layer_activation_function = activation_functions[j],
                        hidden_layer_activation_function_derivative = activation_functions_derivatives[j],
                        output_activation_function = af.softmax,
                        cost_function_derivative = af.cross_entropy_derivative_with_softmax,
                        scaling = True,
                        verbose = True,
                        debug = False)
                    
                    classifier.train_neural_network(
                        learning_rate = learning_rates[j],
                        lambd = 0)

                    scores_tmp[i, j] += classifier.score(classifier.X_test, classifier.y_test)

            return scores_tmp

        parallel = []
        for rep in range(n_repetitions):
            """
            The different processes are created here.
            """
            parallel.append(func.remote())
            # parallel.append(func())

        for res in ray.get(parallel):
            """
            The parallel work is performed and extracted here.
            """
            scores += res
        
        scores /= n_repetitions
        
        np.save(file=fname, arr=scores)

    
    fig, ax = plt.subplots(figsize=(9, 7))
    ax = sns.heatmap(
        data = scores,
        linewidth = 0.5,
        annot = True,
        cmap = 'viridis',
        xticklabels = names,
        yticklabels = hidden_layers,
        ax = ax,
        annot_kws = {"size": 14})

    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(labelsize=15)
    # ax.set_xlabel(r"#  of nodes per layer", fontsize=15)
    ax.set_ylabel(r"# hidden layers", fontsize=15)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.set_ylabel('Accuracy', fontsize=15, rotation=90)
    fig.savefig(dpi=300, fname="../fig/task_d_activations_layers_heatmap.png")
    plt.show()


if __name__ == "__main__":
    # eta_lambda_score_heatmap()
    # nodes_layers_score_heatmap()
    nodes_activations_score_heatmap()
    pass