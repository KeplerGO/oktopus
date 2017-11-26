from abc import abstractmethod
import autograd.numpy as np
from scipy.special import erf
import math


class Model(object):
    def __call__(self, *params):
        return self.evaluate(*params)

    @abstractmethod
    def evaluate(self, *params):
        pass


class ConstantModel(Model):
    def evaluate(self, c):
        return np.array([c])

    def gradient(self, c):
        return [1.]


class LinearModel(Model):
    def __init__(self, X):
        self.X = np.asarray(X)

    def evaluate(self, *theta):
        if len(self.X.shape) > 1:
            w, b = theta[:-1], theta[-1]
            return np.dot(self.X, w) + b
        else:
            w, b = theta
            return self.X * w + b

    def gradient(self, *theta):
        if len(self.X.shape) > 1:
            X_ = [self.X[:, i] for i in range(self.X.shape[-1])]
            return X_ + [np.ones(self.X.shape[0])]
        else:
            return [self.X, np.ones(len(self.X))]


class SymmetricGaussian2D(Model):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def evaluate(self, A, xo, yo, s):
        return A * np.exp(-0.5 * (((self.x - xo) ** 2 + (self.y - yo) ** 2)) / s ** 2)


class Gaussian2D(Model):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def evaluate(self, A, xo, yo, a, b, c):
        return A * np.exp(-(a * (self.x - xo) ** 2
                            - 2 * b * (self.x - xo) * (self.y - yo)
                            + c * (self.y - yo) ** 2))


class Gaussian2DPlusBkg(Gaussian2D):
    def __init__(self, x, y):
        super(Gaussian2DPlusBkg, self).__init__(x, y)

    def evaluate(self, A, xo, yo, a, b, c, B):
        return super(Gaussian2DPlusBkg, self).evaluate(A, xo, yo, a, b, c) + B


class IntegratedSymmetricGaussian2D(Model):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def evaluate(self, F, xo, yo, s):
        return (F / 4 *
                ((erf((self.x - xo + 0.5) / (np.sqrt(2) * s)) -
                  erf((self.x - xo - 0.5) / (np.sqrt(2) * s))) *
                 (erf((self.y - yo + 0.5) / (np.sqrt(2) * s)) -
                  erf((self.y - yo - 0.5) / (np.sqrt(2) * s)))))


class ExpSquaredKernel(Model):
    def __init__(self, t):
        self.t = t

    def evaluate(self, k, l):
        return k * np.exp(- l ** 2 * (self.t[:, None] - self.t[None, :]) ** 2)


class WhiteNoiseKernel(Model):
    def __init__(self, n):
        self.n = n

    def evaluate(self, s):
        return np.diag(np.ones(self.n) * s ** 2)
