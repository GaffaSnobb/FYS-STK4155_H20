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


class CompareNNAndLogistic:
    """
    Generate comparison data of the neural network and logistic
    regression.
    """
    def __init__(self):
        digits = datasets.load_digits()
        self.X = digits.images
        self.y = digits.target
        
        self.n_repetitions = 30
        self.n_categories = 10
        epoch_range = [10, 250]
        self.n_epochs = np.arange(epoch_range[0], epoch_range[1]+1, 5)
        self.n_n_epochs = len(self.n_epochs)
        self.batch_size = 20
        self.learning_rate = 4.8e-3
        self.hidden_layer_sizes = (50,25,25)  # Only for NN.
        self.name = f"_n_repetitions={self.n_repetitions}"
        self.name += f"_n_n_epochs={self.n_n_epochs}"
        self.name += f"_epoch_range=[{epoch_range[0]}_{epoch_range[1]}]"
        self.name += f"_learning_rate={self.learning_rate}"
        self.name += f"_batch_size={self.batch_size}"
        self.name += f"_hidden_layer_sizes={self.hidden_layer_sizes}.npy"

    def generate_data(self, which="both"):
        """
        Generate logistic regression and neural network data based on
        the constructor input parameters.  Save the result as numpy
        binary files.

        Parameters
        ----------
        which : str
            Choose whether to calculate both for NN and LR, or for just
            one.  Allowed inputs are 'both', 'nn', and 'lr'.
        """
        try:
            """
            Check whether the files already exist by trying to import
            them.
            """
            if (which == "both") or (which == "nn"):
                nn_scores = np.load(file="data_files/task_e_nn_scores" + self.name)
            if (which == "both") or (which == "lr"):
                lr_scores = np.load(file="data_files/task_e_lr_scores" + self.name)
            print("Data already generated!")
            print(f"{self.name=}")
            # nn_times = np.load(file="data_files/task_e_nn_times" + name)  # WARNING: TIMING DOES NOT WORK WITH RAY
            # lr_times = np.load(file="data_files/task_e_lr_times" + name)
        
        except FileNotFoundError:
            """
            Generate the data if the files are not found.
            """
            nn_scores = np.zeros(self.n_n_epochs)  # Neural network.
            lr_scores = np.zeros(self.n_n_epochs)  # Logistic regression.
            nn_times = np.zeros(self.n_n_epochs)
            lr_times = np.zeros(self.n_n_epochs)

            ray.init()  
            @ray.remote
            def nn_func():
                """
                This part of the code is put in a function for
                parallelization with ray.

                Returns
                -------
                nn_scores_tmp : numpy.ndarray
                    The neural network scores for the range of number
                    of epochs.

                nn_times_tmp : numpy.ndarray
                    The neural network times for the range of number
                    of epochs.
                """
                nn_classifier = nn.FFNNClassifier(
                    input_data = self.X,
                    true_output = self.y,
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    n_categories = self.n_categories,
                    n_epochs = self.n_epochs,
                    batch_size = self.batch_size,
                    hidden_layer_activation_function = af.sigmoid,
                    hidden_layer_activation_function_derivative = af.sigmoid_derivative,
                    output_activation_function = af.softmax,
                    cost_function_derivative = af.cross_entropy_derivative_with_softmax,
                    scaling = True,
                    verbose = True,
                    debug = False)

                nn_scores_tmp = np.zeros(self.n_n_epochs)
                nn_times_tmp = np.zeros(self.n_n_epochs)

                for i in range(self.n_n_epochs):
                    """
                    Loop over the number of epochs.
                    """
                    print(f"\nrepetition {rep+1} of {self.n_repetitions}")
                    print(f"{i+1} of {self.n_n_epochs}, {self.n_epochs[i]=}")
                    
                    nn_classifier.n_epochs = self.n_epochs[i]
                    nn_classifier.train_neural_network(learning_rate=self.learning_rate)
                    nn_scores_tmp[i] = nn_classifier.score(nn_classifier.X_test, nn_classifier.y_test)
                    nn_times_tmp[i] = nn_classifier.stopwatch

                return nn_scores_tmp, nn_times_tmp
            
            @ray.remote
            def lr_func():
                """
                This part of the code is put in a function for
                parallelization with ray.

                Returns
                -------
                lr_scores_tmp : numpy.ndarray
                    The logistic regression scores for the range of
                    number of epochs.

                nn_times_tmp : numpy.ndarray
                    The logistic regression times for the range of
                    number of epochs.
                """
                lr_classifier = nn.FFNNLogisticRegressor(
                    input_data = self.X,
                    true_output = self.y,
                    hidden_layer_sizes = (),
                    n_categories = self.n_categories,
                    n_epochs = self.n_epochs,
                    batch_size = self.batch_size,
                    output_activation_function = af.softmax,
                    cost_function_derivative = af.cross_entropy_derivative_with_softmax,
                    scaling = True,
                    verbose = True,
                    debug = False)

                lr_scores_tmp = np.zeros(self.n_n_epochs)
                lr_times_tmp = np.zeros(self.n_n_epochs)
                for i in range(self.n_n_epochs):
                    """
                    Loop over the number of epochs.
                    """
                    print(f"\nrepetition {rep+1} of {self.n_repetitions}")
                    print(f"{i+1} of {self.n_n_epochs}, {self.n_epochs[i]=}")
                    
                    lr_classifier.n_epochs = self.n_epochs[i]
                    lr_classifier.train_neural_network(learning_rate=self.learning_rate)
                    lr_scores_tmp[i] = lr_classifier.score(lr_classifier.X_test, lr_classifier.y_test)
                    lr_times_tmp[i] = lr_classifier.stopwatch

                return lr_scores_tmp, lr_times_tmp
            
            if (which == "both") or (which == "nn"):
                """
                Parallelize the repetition of the neural network
                calculations.
                """
                nn_parallel = []
                for rep in range(self.n_repetitions):
                    """
                    The different processes are created here.
                    """
                    nn_parallel.append(nn_func.remote())
                
                for res in ray.get(nn_parallel):
                    """
                    The parallel work is performed and extracted here.
                    """
                    nn_scores_tmp, nn_times_tmp = res
                    nn_scores += nn_scores_tmp
                    nn_times += nn_times_tmp

            if (which == "both") or (which == "lr"):
                """
                Parallelize the repetition of the logistic regression
                calculations.
                """
                lr_parallel = []
                for rep in range(self.n_repetitions):
                    """
                    The different processes are created here.
                    """
                    lr_parallel.append(lr_func.remote())

                for res in ray.get(lr_parallel):
                    """
                    The parallel work is performed and extracted here.
                    """
                    lr_scores_tmp, lr_times_tmp = res
                    lr_scores += lr_scores_tmp
                    lr_times += lr_times_tmp

            # Average the data.                
            nn_scores /= self.n_repetitions
            lr_scores /= self.n_repetitions
            nn_times /= self.n_repetitions
            lr_times /= self.n_repetitions

            if (which == "both") or (which == "nn"):
                np.save(file="data_files/task_e_nn_scores" + self.name, arr=nn_scores)
            if (which == "both") or (which == "lr"):
                np.save(file="data_files/task_e_lr_scores" + self.name, arr=lr_scores)
            # np.save(file="data_files/task_e_nn_times" + self.name, arr=nn_times)   # WARNING: TIMING DOES NOT WORK WITH RAY
            # np.save(file="data_files/task_e_lr_times" + self.name, arr=lr_times)

            print("Complete!")
            print(f"{self.name=}")
    
    def plot_data(self):
        """
        Plot the data with parameters specified in the constructor.
        """
        try:
            nn_scores = np.load(file="data_files/task_e_nn_scores" + self.name)
            lr_scores = np.load(file="data_files/task_e_lr_scores" + self.name)
        except FileNotFoundError:
            print(f"Please generate data before plotting.")
            sys.exit()
        
        fig0, ax0 = plt.subplots(figsize=(9, 7))
        ax0.plot(self.n_epochs, nn_scores, label="Neural network", color="grey")
        ax0.plot(self.n_epochs, lr_scores, label="Logistic regression", color="black", linestyle="dashed")
        ax0.legend(fontsize=15)
        ax0.tick_params(labelsize=15)
        ax0.set_xlabel(r"# epochs", fontsize=15)
        ax0.set_ylabel("Score", fontsize=15)
        ax0.set_title(f"NN max: {np.max(nn_scores):.4f}, LR max: {np.max(lr_scores):.4f}", fontsize=15)
        ax0.grid()
        fig0.savefig(fname="../fig/task_e_nn_lr_score_n_epochs.png" + self.name[:-4] + ".png", dpi=300)
        plt.show()


def plot_specific_files_lr_and_several_nn():
    """
    Import pre-generated data from numpy binaries and plot.  Plot nn
    scores for 1, 2, and 3 hidden layers with scores for logistic
    regression.
    """
    fname0 = "data_files/task_e_nn_scores_n_repetitions=30_n_n_epochs=49_epoch_range=[10_250]_learning_rate=0.0048_batch_size=20_hidden_layer_sizes=(50, 25).npy"
    fname1 = "data_files/task_e_nn_scores_n_repetitions=30_n_n_epochs=49_epoch_range=[10_250]_learning_rate=0.0048_batch_size=20_hidden_layer_sizes=(50,).npy"
    fname2 = "data_files/task_e_lr_scores_n_repetitions=30_n_n_epochs=49_epoch_range=[10_250]_learning_rate=0.0048_batch_size=20_hidden_layer_sizes=(50,).npy"
    fname3 = "data_files/task_e_nn_scores_n_repetitions=30_n_n_epochs=49_epoch_range=[10_250]_learning_rate=0.0048_batch_size=20_hidden_layer_sizes=(50, 25, 25).npy"
    
    nn_scores_two_layers = np.load(fname0)
    nn_scores_one_layer = np.load(fname1)
    nn_scores_three_layers = np.load(fname3)
    lr_scores = np.load(fname2)
    n_epochs = np.arange(10, 250+1, 5)  # This range is inferred from the filenames.
    
    fig0, ax0 = plt.subplots(figsize=(9, 7))
    ax0.plot(n_epochs, nn_scores_one_layer, label="Neural network (50,)",
        color="black")
    ax0.plot(n_epochs, nn_scores_two_layers, label="Neural network (50,25)",
        color="black", linestyle="dashed")
    ax0.plot(n_epochs, nn_scores_three_layers, label="Neural network (50,25,25)",
        color="black", linestyle="dotted")
    ax0.plot(n_epochs, lr_scores, label="Logistic regression", color="maroon")
    ax0.legend(fontsize=15)
    ax0.tick_params(labelsize=15)
    ax0.set_xlabel(r"# epochs", fontsize=15)
    ax0.set_ylabel("Score", fontsize=15)
    ax0.set_title(f"NN max: {np.max([np.max(nn_scores_one_layer), np.max(nn_scores_two_layers)]):.4f}, LR max: {np.max(lr_scores):.4f}", fontsize=15)
    ax0.grid()
    fig0.savefig(fname="../fig/task_e_scores_epochs_n_repetitions=30_n_n_epochs=49_epoch_range=[10_250]_learning_rate=0.0048_batch_size=20_hidden_layer_sizes=(50,25,25).png", dpi=300)
    plt.show()

def plot_specific_files_lr_and_nn_times():
    """
    Import pre-generated data from numpy binaries and plot.  Plot nn and
    lr times as a function of scores.  The scores are a function of
    numbers of epochs.
    """
    fname0 = "data_files/task_e_nn_times_n_repetitions=30_n_n_epochs=49_epoch_range=[10_250]_learning_rate=0.0048_batch_size=20_hidden_layer_sizes=50.npy"
    fname1 = "data_files/task_e_lr_times_n_repetitions=30_n_n_epochs=49_epoch_range=[10_250]_learning_rate=0.0048_batch_size=20_hidden_layer_sizes=50.npy"
    fname2 = "data_files/task_e_nn_scores_n_repetitions=30_n_n_epochs=49_epoch_range=[10_250]_learning_rate=0.0048_batch_size=20_hidden_layer_sizes=(50,).npy"
    fname3 = "data_files/task_e_lr_scores_n_repetitions=30_n_n_epochs=49_epoch_range=[10_250]_learning_rate=0.0048_batch_size=20_hidden_layer_sizes=(50,).npy"

    nn_times = np.load(file=fname0)
    lr_times = np.load(file=fname1)
    nn_scores = np.load(file=fname2)
    lr_scores = np.load(file=fname3)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.plot(nn_times[1:], nn_scores[1:], label="Neural network (50)",
        color="black")
    ax1.plot(lr_times[1:], lr_scores[1:], label="Logistic regression",
        color="maroon")
    ax1.legend(fontsize=15)
    ax1.tick_params(labelsize=15)
    ax1.set_xlabel(r"Time(# batches) [s]", fontsize=15)
    ax1.set_ylabel("Score", fontsize=15)
    ax1.set_title(f"NN max: {np.max(nn_scores):.4f}, LR max: {np.max(lr_scores):.4f}", fontsize=15)
    ax1.grid()
    fig1.savefig(fname="../fig/task_e_nn_lr_score_time.png", dpi=300)
    plt.show()


def compare_logistic_regression_and_neural_network_classification_n_epochs():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    
    n_repetitions = 30
    n_categories = 10
    epoch_range = [10, 250]
    n_epochs = np.arange(epoch_range[0], epoch_range[1]+1, 5)
    n_n_epochs = len(n_epochs)
    batch_size = 20
    learning_rate = 4.8e-3
    hidden_layer_sizes = (50,)  # Only for NN.
    name = f"_n_repetitions={n_repetitions}"
    name += f"_n_n_epochs={n_n_epochs}"
    name += f"_epoch_range=[{epoch_range[0]}_{epoch_range[1]}]"
    name += f"_learning_rate={learning_rate}"
    name += f"_batch_size={batch_size}"
    name += f"_hidden_layer_sizes={hidden_layer_sizes}.npy"

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
            for i in range(n_n_epochs):
                print(f"\nrepetition {rep+1} of {n_repetitions}")
                print(f"{i+1} of {n_n_epochs}, {n_epochs[i]=}")
                
                nn_classifier.n_epochs = n_epochs[i]
                nn_classifier.train_neural_network(learning_rate=learning_rate)
                nn_scores[i] += nn_classifier.score(nn_classifier.X_test, nn_classifier.y_test)
                nn_times[i] += nn_classifier.stopwatch
                
                lr_classifier.n_epochs = n_epochs[i]
                lr_classifier.train_neural_network(learning_rate=learning_rate)
                lr_scores[i] += lr_classifier.score(lr_classifier.X_test, lr_classifier.y_test)
                lr_times[i] += lr_classifier.stopwatch
            
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
    # plt.savefig(fname="../fig/task_e_nn_lr_score_n_epochs.png", dpi=300)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.plot(nn_scores[1:], nn_times[1:], label="Neural network", color="grey")
    ax1.plot(lr_scores[1:], lr_times[1:], label="Logistic regression", color="black", linestyle="dashed")
    ax1.legend(fontsize=15)
    ax1.tick_params(labelsize=15)
    ax1.set_ylabel(r"Time [s]", fontsize=15)
    ax1.set_xlabel("Score", fontsize=15)
    ax1.set_title(f"NN max: {np.max(nn_scores):.4f}, LR max: {np.max(lr_scores):.4f}", fontsize=15)
    # plt.savefig(fname="../fig/task_e_nn_lr_score_time.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # compare_logistic_regression_and_neural_network_classification_learning_rates()
    # compare_logistic_regression_and_neural_network_classification_batch_sizes()
    # compare_logistic_regression_and_neural_network_classification_n_epochs()
    # q = CompareNNAndLogistic()
    # q.generate_data(which="nn")
    plot_specific_files_lr_and_several_nn()
    # plot_specific_files_lr_and_nn_times()
    # q.plot_data()
    pass