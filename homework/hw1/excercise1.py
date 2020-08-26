import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def read_csv():
    filename = "EoS.csv"

    df = pd.read_csv(filename, names=("density", "energy"))
    # df = df.dropna()
    data_entries = len(df.energy)

    X = np.empty((data_entries, 4))     # Design matrix, dim. nxp.

    X[:, 0] = 1     # Intercept.
    X[:, 1] = df.density
    X[:, 2] = df.density**2
    X[:, 3] = df.density**3
    # X[:, 4] = 1

    # plt.plot(df.energy, df.density)
    # plt.show()

    X_train, X_test, y_train, y_test = \
        train_test_split(X, df.energy, test_size=0.2)

    beta = np.linalg.inv(np.transpose(X_train) @ X_train) @ np.transpose(X_train) @ y_train

    # print(df.density)


if __name__ == "__main__":
    read_csv()