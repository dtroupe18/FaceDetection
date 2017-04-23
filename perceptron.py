# a single perceptron
import numpy as np
import matplotlib.pyplot as plt
from random import gauss


class Perceptron(object):
    def __init__(self, rate, epochs, name):
        self.rate = rate
        self.epochs = epochs
        self.name = name

    def fit(self, X, y):
        """
        Fit training data
        :param X: Training vectors, X.shape: [samples, features]
        :param y: Target value, y.shape : [samples]
        :return:
        """

        # generate initial weights with 0 mean and variance 1
        self.weight = [gauss(0, 1) for i in range(1 + X.shape[1])]
        # self.weight = np.zeros(1 + X.shape[1])

        # Number of misclassifications
        self.errors = []

        for i in range(self.epochs):
            err = 0
            for xi, target in zip(X, y):
                delta_w = self.rate * (target - self.predict(xi))
                self.weight[1:] += delta_w * xi
                self.weight[0] += delta_w
                err += int(delta_w != 0)
            self.errors.append(err)
            print("Epoch: ", i + 1, " Number of errors: ", err)
        print("==============================================")
        return self

    def net_input(self, X):
        """ Calculate net input """
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def predict(self, X):
        """ Return class label after unit step """

        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def calculate_results(self, predicted, labels, number_of_samples):
        correct = 0
        incorrect = 0

        for i in range(len(predicted)):
            if predicted[i] == labels[i]:
                correct += 1
            else:
                incorrect += 1

        print(correct, " values were correctly predicted")
        print(100 * (correct / number_of_samples), "% correct for", str(self.name))
        print("______________________________________________")

    def graph_perceptron(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel("Epochs")
        y_label = "Number of misclassifications for ", str(self.name)
        plt.ylabel(y_label)
        plt.show()
