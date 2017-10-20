import autograd.numpy as np
from scipy.special import erf
import math


class Kernel(object):
    pass


class SymmetricGaussian2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def evaluate(self, A, xo, yo, s):
        return A * np.exp(-0.5 * (((self.x - xo) ** 2 + (self.y - yo) ** 2)) / s ** 2)


class Gaussian2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, *params):
        return self.evaluate(*params)

    def evaluate(self, A, xo, yo, a, b, c):
        return A * np.exp(-(a * (self.x - xo) ** 2
                            - 2 * b * (self.x - xo) * (self.y - yo)
                            + c * (self.y - yo) ** 2))


class Gaussian2DPlusBkg(Gaussian2D):
    def __init__(self, x, y):
        super(Gaussian2DPlusBkg, self).__init__(x, y)

    def __call__(self, *params):
        return self.evaluate(*params)

    def evaluate(self, A, xo, yo, a, b, c, B):
        return super(Gaussian2DPlusBkg, self).evaluate(A, xo, yo, a, b, c) + B


class IntegratedSymmetricGaussian2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def evaluate(self, F, xo, yo, s):
        return (F / 4 *
                ((erf((self.x - xo + 0.5) / (np.sqrt(2) * s)) -
                  erf((self.x - xo - 0.5) / (np.sqrt(2) * s))) *
                 (erf((self.y - yo + 0.5) / (np.sqrt(2) * s)) -
                  erf((self.y - yo - 0.5) / (np.sqrt(2) * s)))))


class ExpSquaredKernel(Kernel):
    def __init__(self, t):
        self.t = t

    def evaluate(self, k, l):
        return k * np.exp(- l ** 2 * (self.t[:, None] - self.t[None, :]) ** 2)


class ExpSquaredKernel2D(Kernel):
    def __init__(self, t):
        self.t = t

    def evaluate(self, k, l):
        return k * np.exp(- l ** 2 * np.sum((self.t[:, None] - self.t[None, :]) ** 2, axis=-1))


class WhiteNoiseKernel(Kernel):
    def __init__(self, n):
        self.n = n

    def __call__(self, s):
        return self.evaluate(s)

    def evaluate(self, s):
        return np.diag(np.ones(self.n) * s ** 2)
