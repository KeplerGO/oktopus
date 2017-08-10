import numpy as np

class Gaussian2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_params = 6

    def evaluate(self, A, xo, yo, a, b, c):
        return A * np.exp(- (a * (self.x - xo) ** 2
                             - 2 * b * (self.x - xo) * (self.y - yo)
                             + c * (self.y - yo) ** 2))

class ExpSquaredKernel(object):
    def __init__(self, t):
        self.t = t

    def evaluate(self, k, l):
        return k * np.exp(- np.sum(((self.t[:, None] - self.t[None, :]) / l) ** 2, axis=-1))

class WhiteNoiseKernel(object):
    def __init__(self, n):
        self.n = n

    def evaluate(self, s):
        return np.diag(np.ones(self.n) * s ** 2)
