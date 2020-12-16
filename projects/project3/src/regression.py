import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

def features(degree, n_dependent_variables):
    """
    Calculate the number of features for a given polynomial degree for
    one or two dependent variables.

    Parameters
    ----------
    degree : int
        Polynomial degree.

    n_dependent_variables : int
        The number of dependent variables.
    """
    allowed = [1, 2]
    success = n_dependent_variables in allowed
    msg = "The number of dependent variables must be one of the following:"
    msg += f" {allowed}"
    assert success, msg
    
    if n_dependent_variables == 1:
        return degree + 1
    elif n_dependent_variables == 2:
        return int((degree + 1)*(degree + 2)/2)


def create_design_matrix_one_dependent_variable(x, n_data_total, poly_degree):
    """
    Construct a design matrix with 'n_data_total' rows and
    'poly_degree+1' columns.  For one dependent variable.

    Parameters
    ----------
    x : numpy.ndarray
        Dependent variable.

    n_data_total : int
        The number of data points (rows).

    poly_degree : int
        The polynomial degree (cols-1).

    Returns
    -------
    X : numpy.ndarray
        Design matrix.
    """
    X = np.empty((n_data_total, features(poly_degree, 1)))
    X[:, 0] = 1 # Intercept.

    for i in range(1, poly_degree+1):
        X[:, i] = x**i

    return X


def fit_all_price_data():
    """
    Fit the enire historical BTC price data with linear regression.
    """
    csv_path = "data/btc-usd-max.csv"
    n_data_total = 2774
    polynomial_degrees = [10, 15, 20]
    linestyles = ["dotted", "dashed", "solid"]
    n_polynomial_degrees = len(polynomial_degrees)
    x = np.linspace(0, 1, n_data_total)

    df = pd.read_csv(csv_path, parse_dates=['snapped_at'])
    df = df.sort_values('snapped_at')   # Sort by date.
    price = df.price.values.reshape(-1, 1)  # Reshape to fit the scaler.
    scaler = MinMaxScaler()
    scaled_price = scaler.fit_transform(X = price)
    scaled_price = scaled_price[~np.isnan(scaled_price)] # Remove all NaNs.
    scaled_price = scaled_price.reshape(-1, 1)    # Reshape to column.
    
    fig, ax = plt.subplots(figsize = (9, 7))
    
    for i in range(n_polynomial_degrees):
        
        X = create_design_matrix_one_dependent_variable(x, n_data_total, polynomial_degrees[i])

        X_train, X_test, y_train, y_test = \
            train_test_split(X, scaled_price, test_size = 0.2, shuffle = True)

        reg = LinearRegression().fit(X_train, y_train)
        y_predicted_test = np.sort(X_test, axis = 0)@reg.coef_.T

        ax.plot(
            np.sort(X_test[:, 1])*n_data_total,
            scaler.inverse_transform(y_predicted_test),
            label = f"Poly. deg.: {polynomial_degrees[i]}",
            color = "black",
            linestyle = linestyles[i]
        )

    ax.plot(
        price,
        label = "Actual price",
        color = "gray"
    )
    ax.set_xlabel("Days", fontsize = 15)
    ax.set_ylabel("BTC price", fontsize = 15)
    ax.legend(fontsize = 15)
    ax.tick_params(labelsize = 15)
    ax.grid()
    fig.savefig(fname = "../fig/regression.png", dpi = 300)
    plt.show()


if __name__ == "__main__":
    fit_all_price_data()
    pass