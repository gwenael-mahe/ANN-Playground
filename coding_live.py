import numpy as np


class Perceptron:

    def __init__(self, weights, bias, l_rate):
        self.weights = weights
        self.bias = bias
        self.l_rate = l_rate

    def activation(self, net):
        """ Activation function """
        return np.where(net > 0.5, 1, 0)

    def propagation(self, inputs):
        """ Propagation function"""
        net = np.dot(self.weights, inputs) + self.bias
        return self.activation(net)

    def learning(self, inputs, output):
        """ Learning function"""
        initial_output = self.propagation(inputs)
        self.weights = self.weights + self.l_rate * (initial_output - output) * inputs
        self.bias = self.bias + self.l_rate * (output - initial_output)
        error = np.abs(output - initial_output)
        return error


class Neural:

    def __init__(self):
        self.l_rate = 0.1
