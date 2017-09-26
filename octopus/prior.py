import numpy as np
from .core import LossFunction


__all__ = ['Prior', 'JointPrior', 'UniformPrior', 'GaussianPrior']


class Prior(LossFunction):
    """
    A base class for a prior distribution. Differently from Likelihood, a prior
    is a PDF that depends solely on the parameters, not on the observed data.
    """

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value='param_name'):
        self._name = value


class JointPrior(Prior):
    """Combine indepedent priors by summing the negative of the log
    of their distributions.

    Attributes
    ----------
    args : tuple of instances of Prior

    Examples
    --------
    >>> from octopus import UniformPrior, GaussianPrior, JointPrior
    >>> jp = JointPrior(UniformPrior(-0.5, 0.5), GaussianPrior(0, 1))
    >>> jp.evaluate((0, 0))
    0.0
    >>> jp((0, 0)) # jp is also a callable to .evaluate
    0.0
    """

    def __init__(self, *args):
        self.components = args

    def evaluate(self, params):
        p = 0
        for i in range(len(params)):
            p += self.components[i].evaluate(params[i])
        return p


class UniformPrior(Prior):
    """
    Negative log pdf for a n-dimensional independent uniform distribution.

    Attributes
    ----------
    lb : int or array-like of ints
        Lower bounds (inclusive)
    ub : int or array-like of ints
        Upper bounds (exclusive)

    Examples
    --------
    >>> from octopus import UniformPrior
    >>> unif = UniformPrior(0, 1)
    >>> unif(.5)
    -0.0
    >>> unif(1)
    inf
    """

    def __init__(self, lb, ub, name=None):
        self.lb = np.asarray([lb])
        self.ub = np.asarray([ub])
        self.name = name

    @property
    def mean(self):
        return 0.5 * (self.lb + self.ub)

    @property
    def variance(self):
        return (self.ub - self.lb) ** 2 / 12

    def evaluate(self, params):
        """
        Parameters
        ----------
        params : float, int, or array-like
        """

        if (self.lb <= params).all() and (params < self.ub).all():
            return - np.log(1 / (self.ub - self.lb)).sum()
        return np.inf


class GaussianPrior(Prior):
    """Negative log pdf for a n-dimensional independent Gaussian.

    Attributes
    ----------
    mean : scalar or array-like
        Mean
    var : scalar or array-like
        Variance

    Examples
    --------
    >>> from octopus import GaussianPrior
    >>> gauss = GaussianPrior(0, 1)
    >>> gauss(2)
    2.0
    """

    def __init__(self, mean, var, name=None):
        self.mean = np.asarray([mean])
        self.var = np.asarray([var])
        self.name = name

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = value

    @property
    def variance(self):
        return self.var

    def evaluate(self, params):
        return ((params - self.mean) ** 2 / (2 * self.var)).sum()
