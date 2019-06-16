import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from plotka import plot_decision_regions
import matplotlib.pylab as plt


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    @staticmethod
    def activation(z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return self.activation(self.net_input(X))


class MultiClassPredict(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1, classes=3):
        self.classifiers = []
        self.n_iter = n_iter
        for i in range(classes):
            self.classifiers.append(LogisticRegressionGD(eta, n_iter, random_state))

    def fit(self, X, y):
        for i in range(self.n_iter):
            for i in range(len(self.classifiers)):
                self.classifiers[i].fit(X, y[:, i])

        return self

    def predict(self, X):
        probabilities = []
        for i in range(len(self.classifiers)):
            # pred_classifier = np.reshape(self.classifiers[i].predict(X), (-1, 1))
            probabilities.append(self.classifiers[i].predict(X))
        probabilities = np.array(probabilities)
        probabilities = probabilities.T
        y_predicted = []
        for raw in probabilities:
            y_predicted.append((raw == raw.max()).astype(int))
        y_predicted = np.array(y_predicted)
        predicted_classes = []
        for row in y_predicted:
            predicted_classes.append(np.argmax(row))
        predicted_classes = np.array(predicted_classes)
        return predicted_classes


iris = datasets.load_iris()
X = iris.data[:, [1, 3]]
X = X
y = iris.target

# Standaryzacja zmiennych niezaleznych
sc = StandardScaler()
X_st = sc.fit_transform(X)

# Zakodowanie danych kategorycznych na dummy variables
onehotencoder = OneHotEncoder(categorical_features=[0])
y_st = np.reshape(y, (-1, 1))
y_st = onehotencoder.fit_transform(y_st).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_st, y_st, test_size=0.3, random_state=1, stratify=y_st)

mclass = MultiClassPredict(eta=0.05, n_iter=100, classes=3)
mclass.fit(X_train, y_train)

y_predicted_values = mclass.predict(X_test)
y_test = y_test.dot(onehotencoder.active_features_).astype(int)

probability = y_test[y_test == y_predicted_values].shape[0] / y_test.shape[0]
print(probability)

from sklearn.metrics import confusion_matrix

cnf = confusion_matrix(y_test, y_predicted_values)
print(cnf)

plot_decision_regions(X=X_test, y=y_test, classifier=mclass)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.show()
