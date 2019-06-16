import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Polynomial regression -> using pseudo matrix and mean squares


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


data = pd.read_csv('./Data/dane5.txt', header=None, sep=' ')
data = data.iloc[:, [0, 1]]
X = data.iloc[:, [0]].values.ravel()
y = data.iloc[:, [1]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)

degrees = np.linspace(1, 11, 11, dtype=int)

root_squares = []
mean_squares = []

for degree in degrees:
    matrix = np.array([regression_order(degree, n) for n in X_train])
    pseudo_matrix = np.linalg.pinv(matrix)

    weights = pseudo_matrix @ y_train

    plt.plot(X, y, 'go')
    plt.plot(X, poly_model(X, weights), 'bo')
    plt.title("Plot with {} degree".format(degree))
    plt.show()

    y_predicted = poly_model(X_test, weights)
    poly_mse = mean_squared_error(y_test, y_predicted)

    SS_Residual = sum((y_test - y_predicted) ** 2)
    SS_Total = sum((y_test - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - 1 - 1)
    print("R2: {}, AdjR2: {}".format(r_squared, adjusted_r_squared))

    mean_squares.append(poly_mse)
    poly_rmse = np.sqrt(poly_mse)

    root_squares.append(poly_rmse)

root_squares = np.array(root_squares)
mean_squares = np.array(mean_squares)
best_degree = root_squares.argmin() + 1

print('Best degree: {} with root mean squared error {}'.format(best_degree, root_squares.min()))

# Linear regression using LS function


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('./Data/dane1.txt', header=None, sep=' ')
data = data.iloc[:, [0, 1]]
X = data.iloc[:, [0]].values.ravel()
y = data.iloc[:, [1]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)


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


def fun_2(x, a, b, c):
    return x ** 2 * a + x * b + c


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


b0, b1 = linear_model(X_train, y_train)

y_predicted = fun(X_test, b0, b1)
poly_mse = mean_squared_error(y_test, y_predicted)
print(poly_mse)


y_predicted = fun(X_train, b0, b1)
poly_mse = mean_squared_error(y_train, y_predicted)
print(poly_mse)

plt.plot(X_train, y_train, 'go')
plt.plot(X_train, fun(X_train, b0, b1), 'b-')
plt.show()

plt.plot(X_test, y_test, 'go')
plt.plot(X_test, fun(X_test, b0, b1), 'bo')
plt.show()

matrix = np.array([[n ** 2, n, 1] for n in X_train])
pseudo_matrix = np.linalg.pinv(matrix)

weights = pseudo_matrix @ y_train

y_predicted = poly_model(X_train, weights)
poly_mse = mean_squared_error(y_train, y_predicted)
print(poly_mse)

y_predicted = poly_model(X_test, weights)
poly_mse = mean_squared_error(y_test, y_predicted)
print(poly_mse)

plt.plot(X, y, 'go')
plt.plot(X, fun_2(X, *weights), 'bo')
plt.show()
