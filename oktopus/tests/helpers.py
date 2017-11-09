# Define simple models to be used during tests
import autograd.numpy as npa


class ConstantModel(object):
    def __call__(self, c):
        return self.evaluate(c)

    def evaluate(self, c):
        return npa.array([c])

    def gradient(self, c):
        return [1.]


class LineModel(object):
    def __init__(self, x):
        self.x = x

    def __call__(self, m, b):
        return self.evaluate(m, b)

    def evaluate(self, m, b):
        return m * self.x + b

    def gradient(self, m, b):
        return [self.x, npa.ones(len(self.x))]
