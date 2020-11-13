import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set()
np.random.seed(0)

digits = datasets.load_digits()
inputs = digits.images
labels = digits.target

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))

# flatten the image
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, shuffle=True)

def hiddenlayer(n_nodes, n_hiddenlayers):
    """
    Want the same number of nodes per hdden layer,
    so makes a tuple of numbers of nodes, one for each 
    hidden layer.
    """
    h = []
    for i in range(n_hiddenlayers):
        h.append(n_nodes)
    h = tuple(i for i in h)
    return h

def accuracy_score(Y_test, Y_pred):
    """
    Calculate accuracy score to measure the performance 
    of the network. 
    """
    return np.sum(Y_test == Y_pred) / len(Y_test)
"""
# Setting parameters
hiddenlayers = np.arange(1,4,1)
n_hiddenlayers = len(hiddenlayers)
nodes = np.arange(8,12,1)
n_nodes = len(nodes)
n_lambdas = 5
lambdas = np.logspace(-5, -1, n_lambdas)
n_step_sizes = 4
step_sizes = np.linspace(1e-4, 1e-1, n_step_sizes)
af = ['identity', 'logistic', 'tanh', 'relu']
n_a = len(af)

accuracies = np.zeros((n_hiddenlayers,n_nodes,n_step_sizes,n_lambdas,n_a))

# Looping over number of hidden layers, nodes, steps and lambda-values
for h in range(n_hiddenlayers):
    for n in range(n_nodes):
        for s in range(n_step_sizes):
            for l in range(n_lambdas):
                for a in range(n_a):
                    im = MLPClassifier(hidden_layer_sizes=hiddenlayer(nodes[n], hiddenlayers[h]), 
                    activation=af[a], solver='sgd', alpha=lambdas[l], 
                    batch_size='auto', learning_rate='constant', learning_rate_init=step_sizes[s], 
                    max_iter=200)
                    im.fit(X_train, y_train)
                    test_predict = im.predict(X_test)
                    accuracies[h,n,s,l,a] = accuracy_score(y_test, test_predict)

# Finding the greatest accuracy score
idx = np.unravel_index(np.argmax(accuracies), accuracies.shape)
print("The output activation function is %s." %im.out_activation_)
print("Accuracy score on test set: %g with %d hidden layer, %d nodes, %g step_size, %g lambda and the %s activation function." %(np.max(accuracies),hiddenlayers[idx[0]],nodes[idx[1]],step_sizes[idx[2]],lambdas[idx[3]],af[idx[4]]))

t_lambdas = tuple('%g' %i for i in lambdas)
t_step_sizes = tuple('%.2f' %i for i in step_sizes)
t_hiddenlayers = tuple('%d' %i for i in hiddenlayers)
t_nodes = tuple('%d' %i for i in nodes)

# Plotting the accuracy score vs all parameters
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(accuracies[idx[0],idx[1],:,:,idx[4]], annot=True, ax=ax, cmap="viridis", cbar_kws={'label':'Accuracy score'})
plt.xticks(np.arange(.5,n_lambdas+.5), t_lambdas)
plt.yticks(np.arange(.5,n_step_sizes+.5), t_step_sizes)
ax.set_title("Test Accuracy")
ax.set_xlabel("$\lambda$")
ax.set_ylabel("$\eta$")
plt.savefig(dpi=300, fname="task_d_accuracy_lambdas_steps")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(accuracies[:,idx[1],idx[2],idx[3],:], annot=True, ax=ax, cmap="viridis", cbar_kws={'label':'Accuracy score'})
plt.xticks(np.arange(.5,n_a+.5),['identity', 'logistic', 'tanh', 'relu'],rotation=20)
plt.yticks(np.arange(.5,n_hiddenlayers+.5), t_hiddenlayers)
ax.set_title("Test Accuracy")
ax.set_xlabel("Activation function")
ax.set_ylabel("Number of hidden layers")
plt.savefig(dpi=300, fname="task_d_accuracy_hiddenlayers_activation")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(accuracies[:,:,idx[2],idx[3],idx[4]], annot=True, ax=ax, cmap="viridis", cbar_kws={'label':'Accuracy score'})
plt.xticks(np.arange(.5,n_nodes+.5),t_nodes)
plt.yticks(np.arange(.5,n_hiddenlayers+.5), t_hiddenlayers)
ax.set_title("Test Accuracy")
ax.set_ylabel("Number of hidden layers")
ax.set_xlabel("Number of nodes")
plt.savefig(dpi=300, fname="task_d_accuracy_hiddenlayers_nodes")
plt.show()
"""
hiddenlayers = np.arange(1,4,1)
n_hiddenlayers = len(hiddenlayers)
nodes = np.arange(8,12,1)
n_nodes = len(nodes)
n_lambdas = 5
lambdas = np.logspace(-8, -4, n_lambdas)
n_step_sizes = 4
step_sizes = np.linspace(1e-4, 1e-1, n_step_sizes)
af = ['identity', 'logistic', 'tanh', 'relu']
n_a = len(af)
accuracies = np.zeros((n_hiddenlayers,n_nodes,n_lambdas,n_a))

# Looping over number of hidden layers, nodes, steps and lambda-values
for h in range(n_hiddenlayers):
    for n in range(n_nodes):
        for l in range(n_lambdas):
            for a in range(n_a):
                im = MLPClassifier(hidden_layer_sizes=hiddenlayer(nodes[n], hiddenlayers[h]), 
                activation=af[a], solver='sgd', alpha=lambdas[l], 
                batch_size='auto', learning_rate='invscaling', 
                max_iter=200)
                im.fit(X_train, y_train)
                test_predict = im.predict(X_test)
                accuracies[h,n,l,a] = accuracy_score(y_test, test_predict)

idx = np.unravel_index(np.argmax(accuracies), accuracies.shape)
print("The output activation function is %s." %im.out_activation_)
print("Accuracy score on test set: %g with %d hidden layer, %d nodes, %g lambda and the %s activation function." %(np.max(accuracies),hiddenlayers[idx[0]],nodes[idx[1]],lambdas[idx[2]],af[idx[3]]))
t_lambdas = tuple('%g' %i for i in lambdas)
t_step_sizes = tuple('%.2f' %i for i in step_sizes)
t_hiddenlayers = tuple('%d' %i for i in hiddenlayers)
t_nodes = tuple('%d' %i for i in nodes)
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(accuracies[idx[0],idx[1],:,:], annot=True, ax=ax, cmap="viridis", cbar_kws={'label':'Accuracy score'})
plt.xticks(np.arange(.5,n_a+.5),['identity', 'logistic', 'tanh', 'relu'],rotation=20)
plt.yticks(np.arange(.5,n_lambdas+.5), t_lambdas)
ax.set_title("Test Accuracy")
ax.set_xlabel("Activation function")
ax.set_ylabel("$\lambda$")
plt.savefig(dpi=300, fname="task_d_accuracy_hiddenlayers_activation_l")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(accuracies[:,:,idx[2],idx[3]], annot=True, ax=ax, cmap="viridis", cbar_kws={'label':'Accuracy score'})
plt.xticks(np.arange(.5,n_nodes+.5),t_nodes)
plt.yticks(np.arange(.5,n_hiddenlayers+.5), t_hiddenlayers)
ax.set_title("Test Accuracy")
ax.set_ylabel("Number of hidden layers")
ax.set_xlabel("Number of nodes")
plt.savefig(dpi=300, fname="task_d_accuracy_hiddenlayers_nodes_l")
plt.show()

"""Output:
Accuracy score on test set: 0.969444 with 1 hidden layer, 9 nodes, 
0.0334 step_size, 0.001 lambda and the logistic activation function.
Different learning rate scheme: Accuracy score on test set: 
0.544444 with 1 hidden layer, 10 nodes, 
0.0001 lambda and the identity activation function.

"""

