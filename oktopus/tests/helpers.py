# Define simple models to be used during tests
import autograd.numpy as npa


class Model(object):
    def __call__(self, *params):
        return self.evaluate(*params)


class ConstantModel(Model):
    def evaluate(self, c):
        return npa.array([c])

    def gradient(self, c):
        return [1.]


class LineModel(Model):
    def __init__(self, x):
        self.x = x

    def evaluate(self, m, b):
        return m * self.x + b

    def gradient(self, m, b):
        return [self.x, npa.ones(len(self.x))]
