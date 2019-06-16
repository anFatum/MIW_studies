import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


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


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
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

    n_iter = 50
    eta = 0.05
    random_state = 1
    classes = 3

    classifiers = []
    for i in range(classes):
        classifiers.append(LogisticRegressionGD(eta, n_iter, random_state))

    for i in range(n_iter):
        for n in range(classes):
            classifiers[n].fit(X_train, y_train[:, n])

    probabilities = []
    for i in range(classes):
        probabilities.append(classifiers[i].predict(X_test))

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

    y_test = y_test.dot(onehotencoder.active_features_).astype(int)

    probability = y_test[y_test == predicted_classes].shape[0] / y_test.shape[0]
    print(probability)


if __name__ == "__main__":
    main()
