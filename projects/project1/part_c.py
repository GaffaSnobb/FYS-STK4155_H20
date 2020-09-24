import numpy as np
import matplotlib.pyplot as plt
from common import create_design_matrix


def franke(x, y):
    return 3/4*np.exp(-(9*x - 2)**2/4 - (9*y - 2)**2/4) + 3/4*np.exp(-(9*x + 1)**2/49 - (9*y + 1)/10) + 1/2*np.exp(-(9*x - 7)**2/4 - (9*y - 3)**2/4) - 1/5*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)


def cross_validation(X, y, folds):
    rest = X.shape[0]%folds
    X = X[0:X.shape[0]-rest]    # Remove the rest to get equally sized folds.
    y = y[0:y.shape[0]-rest] # Remove the rest to get equally sized folds.

    mse = 0
    
    for i in range(folds):
        y_split = np.split(y, folds)
        y_validation = y_split.pop(i)
        y_training = np.concatenate(y_split)

        X_split = np.split(X, folds)
        X_validation = X_split.pop(i)
        X_training = np.concatenate(X_split)

        beta = np.linalg.pinv(X_training.T@X_training)@X_training.T@y_training

        y_predicted = X_validation@beta
        mse += np.mean((y_predicted - y_validation)**2)

    mse /= folds

    return mse

if __name__ == "__main__":
    N = 800
    degrees = np.arange(1, 20+1, 1)
    N_degrees = len(degrees)
    noise_factor = 0.1
    folds = 5
    x1 = np.random.uniform(0, 1, N)
    x2 = np.random.uniform(0, 1, N)

    y_observed = franke(x1, x2) + noise_factor*np.random.randn(N)
    mse = np.empty(N_degrees)

    for i in range(N_degrees):
        X = create_design_matrix(x1=x1, x2=x2, N=N, deg=degrees[i])
        mse[i] = cross_validation(X, y_observed, folds)

    plt.plot(degrees, mse)
    plt.show()


