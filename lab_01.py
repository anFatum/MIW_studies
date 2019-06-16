import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Zwraca tablice poteg x do n [x^n, x^(n-1) ...]
def return_degree_array(n, x):
    model = []
    for i in range(n, -1, -1):
        model.append(x ** i)
    return model


# Zwraca wyjscie regresji a*x^n + b*x^(n-1) + ...
def get_polynomial(x, weights):
    sum = 0
    for i in range(len(weights)):
        ind = len(weights) - i - 1
        sum += x ** i * weights[ind]
    return sum

# Zwraca wspolczynniki dla regresji liniowej za pomoca least squared error
def linear_model(X, Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    x_mean_distance = np.array([x - x_mean for x in X])
    x_mean_squared_distance = np.array([(x - x_mean) ** 2 for x in X])
    y_mean_distance = np.array([y - y_mean for y in Y])

    x_sum_squared_distance = x_mean_squared_distance.sum()
    sum_distance = (x_mean_distance * y_mean_distance).sum()

    b1 = sum_distance / x_sum_squared_distance

    b0 = x_mean * b1 - y_mean

    return b0, b1


def fun(x, n, m):
    return n + m * x


def main():
    data = pd.read_csv('./Data/dane7.txt', header=None, sep=' ')
    data = data.iloc[:, [0, 1]]
    X = data.iloc[:, [0]].values.ravel()
    y = data.iloc[:, [1]].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=1)

    b0, b1 = linear_model(X_train, y_train)

    y_predicted = fun(X_test, b0, b1)
    poly_mse = mean_squared_error(y_test, y_predicted)
    print(poly_mse)

    y_predicted = fun(X_train, b0, b1)
    poly_mse = mean_squared_error(y_train, y_predicted)
    print(poly_mse)
    print()

    plt.plot(X_train, y_train, 'go')
    plt.plot(X_train, fun(X_train, b0, b1), 'b-')
    plt.show()

    matrix = np.array([return_degree_array(17, n) for n in X_train])
    pseudo_matrix = np.linalg.pinv(matrix)

    weights = pseudo_matrix @ y_train

    plt.plot(X, y, 'go')
    plt.plot(X, get_polynomial(X, weights), 'b-')
    plt.title(f"Plot with {17} degree")
    plt.show()

    y_predicted = get_polynomial(X_test, weights)
    poly_mse = mean_squared_error(y_test, y_predicted)
    print(poly_mse)

    y_predicted = get_polynomial(X_train, weights)
    poly_mse = mean_squared_error(y_train, y_predicted)
    print(poly_mse)


if __name__ == '__main__':
    main()
