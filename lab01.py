import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def regression_order(n, x):
    model = []
    for i in range(n, -1, -1):
        model.append(x ** i)
    return np.array(model)


def poly_model(x, weights):
    sum = 0
    for i in range(len(weights)):
        ind = len(weights) - i - 1
        sum += x ** i * weights[ind]
    return sum


"""
Using least squared method for evaluating parameters for regression models
Linear model has the form ax + b
Polynomial degree was calculated in process of experiments (with use of 
mean squared errors) and has the form 
ax^7 + bx^6 + cx^5 + ...
Least squared method -> calculating pseudo matrix for independent variable x and its degrees
"""


def main():
    data = pd.read_csv('./Data/dane5.txt', header=None, sep=' ')
    data = data.iloc[:, [0, 1]]
    X = data.iloc[:, [0]].values.ravel()
    y = data.iloc[:, [1]].values.ravel()

    print(f'Min value for x: {X.min()}, max value for x: {X.max()}')
    print(f'Min value for y: {y.min()}, max value for y: {y.max()}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=1)

    matrix = np.array([regression_order(1, n) for n in X_train])
    pseudo_matrix = np.linalg.pinv(matrix)

    weights = pseudo_matrix @ y_train  # matrix multiplicity

    plt.plot(X, y, 'go')
    plt.plot(X, poly_model(X, weights), 'bo')
    plt.title(f"Plot with {1} degree")
    plt.show()

    y_predicted = poly_model(X_test, weights)
    poly_mse = mean_squared_error(y_test, y_predicted)

    print(f'Root squared error for linear model: {poly_mse}')

    matrix = np.array([regression_order(7, n) for n in X_train])
    pseudo_matrix = np.linalg.pinv(matrix)

    weights = pseudo_matrix @ y_train

    plt.plot(X, y, 'go')
    plt.plot(X, poly_model(X, weights), 'bo')
    plt.title(f"Plot with {7} degree")
    plt.show()

    y_predicted = poly_model(X_test, weights)
    poly_mse = mean_squared_error(y_test, y_predicted)

    print(f'Root squared error for polynomial model: {poly_mse}')


if __name__ == '__main__':
    main()
