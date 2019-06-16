import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)
        # return np.where(self.net_input(X) >= 0.0, 1, -1)


class MultiClassPredict(object):
    def __init__(self, eta=0.01, n_iter=10, classes=3):
        self.classifiers = []
        self.n_iter = n_iter
        for i in range(classes):
            self.classifiers.append(Perceptron(eta, n_iter))

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

    # def main():


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

sc = StandardScaler()
X_st = sc.fit_transform(X)

# Zakodowanie danych kategorycznych na dummy variables
onehotencoder = OneHotEncoder(categorical_features=[0])
y_st = np.reshape(y, (-1, 1))
y_st = onehotencoder.fit_transform(y_st).toarray()
# w perceptronie wyjście jest albo 1 albo -1
# y_train_01_subset[(y_train_01_subset == 0)] = -1
y_st[y_st == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X_st, y_st, test_size=0.3, random_state=1, stratify=y)

mclass = MultiClassPredict(eta=0.05, n_iter=10, classes=3)
mclass.fit(X_train, y_train)

predicted = mclass.predict(X_test)

y_test[y_test == -1] = 0
y_test = y_test.dot(onehotencoder.active_features_).astype(int)

probability = y_test[y_test == predicted].shape[0] / y_test.shape[0]
print(probability)

from sklearn.metrics import confusion_matrix

cnf = confusion_matrix(y_test, predicted)
print(cnf)

plot_decision_regions(X=X_test, y=y_test, classifier=mclass)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.show()
