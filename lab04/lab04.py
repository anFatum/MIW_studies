import math

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lab04.ANLayer import ANLayer


def ReLU(x):
    mask = (x > 0) * 1.0
    return mask * x


def d_ReLU(x):
    return np.exp(x) / (np.exp(x) + 1)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def d_sigmoid(X):
    return X * (1 - X)


def linear(X):
    return X


def d_linear(X):
    return 1


def tanh(X):
    return np.tanh(X)


def d_tanh(X):
    return 1 - np.power(X, 2)


learning_rate = 0.001

data = pd.read_csv('./lab04/Data/dane4.txt', header=None, sep=' ')
data = data.iloc[:, [0, 1]]

X = data.iloc[:, [0]].values

y = data.iloc[:, [1]].values
y = np.reshape(y, (-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)

an_one = ANLayer(150, 1, training='online')
an_two = ANLayer(1, 150, connected=an_one.neurons, training='online')

# an_one = ANLayer(100, 1)
# an_two = ANLayer(1, 100, connected=an_one.neurons)


for i in range(200):
    an_one_outs = np.array(list(an_one.get_outs(X_train, tanh)))
    an_two.back_propagate(learning_rate, an_one_outs, linear, d_linear, y=y_train)
    an_one.back_propagate(learning_rate, X_train, tanh, d_tanh)
    an_one_outs = np.array(list(an_one.get_outs(X_train, tanh)))
    an_two_outs = np.array(list(an_two.get_outs(an_one_outs, linear)))
    if i % 3 == 0:
        plt.plot(X_train, y_train, 'go')
        plt.plot(X_train, np.array(list(an_two.get_outs(an_one_outs, linear))), 'bo')
        plt.show()

# for i in range(200):
#     an_one_outs = np.array(list(an_one.get_outs(X_train, ReLU)))
#     an_two.back_propagate(learning_rate, an_one_outs, linear, d_linear, y=y_train)
#     an_one.back_propagate(learning_rate, X_train, ReLU, d_ReLU)
#     an_one_outs = np.array(list(an_one.get_outs(X_train, ReLU)))
#     an_two_outs = np.array(list(an_two.get_outs(an_one_outs, linear)))
#     if i % 3 == 0:
#         plt.plot(X_train, y_train, 'go')
#         plt.plot(X_train, np.array(list(an_two.get_outs(an_one_outs, linear))), 'bo')
#         plt.show()

learning_rate = 0.001

data = pd.read_csv('./lab04/Data/dane4.txt', header=None, sep=' ')
data = data.iloc[:, [0, 1]]

X = data.iloc[:, [0]].values

y = data.iloc[:, [1]].values
y = np.reshape(y, (-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)


an_one = ANLayer(90, 1, training='online')
an_two = ANLayer(25, 90, connected=an_one.neurons, training='online')
an_three = ANLayer(1, 25, connected=an_two.neurons, training='online')

for i in range(80):
    an_one_outs = np.array(list(an_one.get_outs(X_train, tanh)))
    an_two_outs = np.array(list(an_two.get_outs(an_one_outs, ReLU)))
    an_three.back_propagate(learning_rate, an_two_outs, linear, d_linear, y=y_train)
    an_two.back_propagate(learning_rate, an_one_outs, ReLU, d_ReLU)
    an_one.back_propagate(learning_rate, X_train, tanh, d_tanh)
    an_one_outs = np.array(list(an_one.get_outs(X_train, tanh)))
    an_two_outs = np.array(list(an_two.get_outs(an_one_outs, ReLU)))
    an_three_outs = np.array(list(an_three.get_outs(an_two_outs, linear)))
    if i % 3 == 0:
        plt.plot(X_train, y_train, 'go')
        plt.plot(X_train, an_three_outs, 'bo')
        plt.show()
