import numpy as np
from abc import ABC, abstractmethod
from scipy.special import erf
import math

class Kernel(ABC):
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

    def __call__(self, A, xo, yo, a, b, c):
        return self.evaluate(A, xo yo, a, b, c)

    def evaluate(self, A, xo, yo, a, b, c):
        return A * np.exp(-(a * (self.x - xo) ** 2
                            - 2 * b * (self.x - xo) * (self.y - yo)
                            + c * (self.y - yo) ** 2))


class Gaussian2DPlusBkg(Gaussian2D):
    def __init__(self, x, y):
        super(Gaussian2DPlusBkg, self).__init__(x, y)

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

    def evaluate(self, s):
        return np.diag(np.ones(self.n) * s ** 2)


def get_initial_guesses(data):
    """
    Compute the initial guess for PSF width using the sample moments of
    the data.

    Parameters
    ----------
    data : 2D array-like
        Image data.

    Return
    ------
    sigma : float
        Initial guess for the width of the PSF.
    """

    total = data.sum()
    Y, X = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total

    marg_x = data[:, int(x)]
    marg_y = data[int(y), :]

    sigma_y = math.sqrt(np.abs((np.arange(marg_y.size) - y) ** 2 * marg_y).sum() / marg_y.sum())
    sigma_x = math.sqrt(np.abs((np.arange(marg_x.size) - x) ** 2 * marg_x).sum() / marg_x.sum())
    sigma = math.sqrt((sigma_x**2 + sigma_y**2)/2.0)

    return total, x, y, sigma
