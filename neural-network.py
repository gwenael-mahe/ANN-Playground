import numpy as np


class ANN:

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.weights = self.get_weights()
        self.bias = 0.5
        self.l_rate = 0.1
        self.epoch = 20
        self.epoch_loss = []

    def get_weights(self):
        rg = np.random.default_rng()
        weights = rg.random((1, len(self.features) - 1))[0]
        return weights

    def get_weighted_sum(self):
        return np.dot(self.features, self.weights) + self.bias

    def sigmoid(self, w_sum):
        return 1 / (1 + np.exp(-w_sum))

    def cross_entropy(self, target, prediction):
        return -(target * np.log10(prediction) + (1 - target * np.log10(1 - prediction)))

    def update_weights(self, target, prediction, feature):
        new_weights = []
        for x, w in zip(feature, self.weights):
            new_w = w + self.l_rate * (target - prediction) * x
            new_weights.append(new_w)
        return new_weights

    def update_bias(self, target, prediction):
        self.bias = self.bias + self.l_rate * (target - prediction)

    def fit(self):
        individual_loss = []
        for e in range(self.epoch):
            for i in range(len(self.features)):
                w_sum = self.get_weighted_sum()
                prediction = self.sigmoid(w_sum)
                loss = self.cross_entropy(self.target, prediction)
                individual_loss.append(loss)
                # gradient descent
                self.weights = self.update_weights(self.target, prediction, self.features)
                self.update_bias(self.target, prediction)
            avg_loss = sum(individual_loss) / len(individual_loss)
            self.epoch_loss.append(avg_loss)
            print("epoch: ", e)
            print("avg loss: ", avg_loss)
