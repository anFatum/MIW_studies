import numpy as np
import math


def ReLU(x):
    mask = (x > 0) * 1.0
    return mask * x


def d_ReLU(x):
    mask = (x > 0) * 1.0
    return mask


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def d_sigmoid(X):
    return X * (1 - X)


class ANLayer:
    def __init__(self, num_neurons, num_wages, training='batch', connected=None):
        if training not in {'batch', 'online'}:
            raise ValueError('No such training method')
        self.training = training
        self.num_neurons = num_neurons
        self.num_wages = num_wages
        self.connected = connected
        neurons = []
        for neu in range(num_neurons):
            wages = []
            for wage in range(num_wages):
                wages.append(np.random.uniform(-1, 1))
            b = np.random.uniform(-1, 1)
            error = 0.
            neurons.append([np.array(wages), b, error])
        self.neurons = np.array(neurons)

    def get_outs(self, X, activation_function=ReLU):
        if X.ndim == 1:
            neuron_res = []
            for neuron in self.neurons:
                activation = np.sum(neuron[0].dot(X), dtype=float)
                activation += neuron[1]
                o = activation_function(activation)
                neuron_res.append(o)
            return np.array(neuron_res)
        res = []
        for data in X:
            neuron_res = []
            for neuron in self.neurons:
                activation = np.sum(neuron[0].dot(data), dtype=float)
                activation += neuron[1]
                o = activation_function(activation)
                neuron_res.append(o)
            # print(res)
            res.append(neuron_res)
        return np.array(res)

    def back_propagate(self, learning_rate, inp, activation_function=ReLU, derivative_function=d_ReLU,
                       y=None):
        if self.training == 'batch':
            out = self.get_outs(inp, activation_function)
            if y is not None:
                self.back_propagate_outer_layer(learning_rate, inp,
                                                derivative_function, y=y, out=out)
            else:
                self.back_propagate_inner_layer(learning_rate, inp, out, derivative_function)
        else:
            ind = 0
            for raw in inp:
                out = self.get_outs(raw, activation_function)
                if y is not None:
                    self.back_propagate_outer_layer(learning_rate, inp[ind],
                                                    derivative_function, y=y[ind], out=np.array(out))
                else:
                    self.back_propagate_inner_layer(learning_rate, inp[ind], np.array(out), derivative_function)
                ind += 1

    def back_propagate_outer_layer(self, learning_rate, inp, derivative_function, y, out):
        delta = y - out
        error = learning_rate * delta * derivative_function(out)
        error_to_update_wages = error if error.ndim != 1 else np.reshape(error, (1, -1))
        reshaped_x = inp if inp.ndim != 1 else np.reshape(inp, (1, -1))
        error_to_update_wages = error_to_update_wages.T.dot(reshaped_x)
        error = error.sum(axis=0)
        error_to_propagate = []
        for i in range(len(self.neurons)):
            error_to_propagate.append(np.array(self.neurons[:, 0][i]) *
                                      error if (isinstance(error, np.float) or isinstance(error, np.int))
                                      else error[i])
        for i in range(len(self.neurons)):
            self.neurons[i][0] = self.neurons[i][0] + error_to_update_wages[i].T
            self.neurons[i][1] = self.neurons[i][1] + error if (
                    isinstance(error, np.float) or isinstance(error, np.int)) else error[i]
            self.connected[:, 2] = self.connected[:, 2] + np.array(error_to_propagate)

    def back_propagate_inner_layer(self, learning_rate, inp, out, derivative_function):
        error = self.neurons[:, 2]
        error = error * learning_rate * derivative_function(out)
        error_to_update_wages = error if error.ndim != 1 else np.reshape(error, (1, -1))
        reshaped_x = inp if inp.ndim != 1 else np.reshape(inp, (1, -1))
        error_to_update_wages = error_to_update_wages.T.dot(reshaped_x)
        error = error.sum(axis=0)
        error = np.minimum(error, math.factorial(21))
        error_to_propagate = []
        for i in range(len(self.neurons)):
            error_to_propagate.append(np.array(self.neurons[:, 0][i]) *
                                      error if (isinstance(error, np.float) or isinstance(error, np.int))
                                      else error[i])
        for i in range(len(self.neurons)):
            self.neurons[i][0] = self.neurons[i][0] + error_to_update_wages[i].T
            self.neurons[i][1] = self.neurons[i][1] + error if \
                (isinstance(error, np.float) or isinstance(error, np.int)) else error[i]
