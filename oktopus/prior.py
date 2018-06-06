from abc import abstractmethod
import autograd.numpy as np
from .loss import LossFunction


__all__ = ['Prior', 'JointPrior', 'UniformPrior', 'GaussianPrior', 'LaplacianPrior']


class Prior(LossFunction):
    """
    A base class for a prior distribution. Differently from Likelihood, a prior
    is a PDF that depends solely on the parameters, not on the observed data.
    """

    @property
    def name(self):
        """A name associated with the prior"""
        return self._name

    @name.setter
    def name(self, value='param_name'):
        self._name = value

    @abstractmethod
    def evaluate(self, params):
        """Evaluates the negative of the log of the PDF at ``params``

        Parameters
        ----------
        params : scalar or array-like
            Value at which the PDF will be evaluated

        Returns
        -------
        value : scalar
            Value of the negative of the log of the PDF at params
        """
        pass


class JointPrior(Prior):
    """Combine indepedent priors by summing the negative of the log
    of their distributions.

    Attributes
    ----------
    *args : tuple of instances of :class:`Prior`
        Instances of :class:`Prior` to be combined

    Examples
    --------
    >>> from oktopus import UniformPrior, GaussianPrior, JointPrior
    >>> jp = JointPrior(UniformPrior(-0.5, 0.5), GaussianPrior(0., 1.))
    >>> jp.evaluate((0., 0.))
    0.0
    >>> jp((0., 0.)) # jp is also a callable to evaluate()
    0.0

    Notes
    -----
    *args are stored in ``self.components``.
    """

    def __init__(self, *args):
        self.components = args

    def __repr__(self):
        return "<JointPrior({})>".format([c.__repr__() for c in self.components])

    def evaluate(self, params):
        """Computes the sum of the log of each distribution given in *args*
        evaluated at *params*.

        Parameters
        ----------
        params : tuple
            Value at which the JointPrior instance will be evaluated.
            This must have the same dimension as the number of priors used
            to initialize the object

        Returns
        -------
        value : scalar
            Sum of the negative of the log of each distribution given in **args**
        """
        p = 0
        for i in range(len(params)):
            p += self.components[i].evaluate(params[i])
        return p

    def gradient(self, params):
        """Computes the gradient of the sum of the log of each distribution
        given in *args* evaluated at *params*.

        Parameters
        ----------
        params : tuple
            Value at which the JointPrior instance will be evaluated.
            This must have the same dimension as the number of priors used
            to initialize the object

        Returns
        -------
        value : scalar
            Gradient of the sum of the negative of the log of each distribution
            given in **args**
        """
        grad = 0
        for i in range(len(params)):
            grad += self.components[i].gradient(params[i])
        return grad

    @property
    def mean(self):
        return np.concatenate([self.components[i].mean for i in range(len(self.components))])


class UniformPrior(Prior):
    """Computes the negative log pdf for a n-dimensional independent uniform
    distribution.

    Attributes
    ----------
    lb : int or array-like of ints
        Lower bounds (inclusive)
    ub : int or array-like of ints
        Upper bounds (exclusive)

    Examples
    --------
    >>> from oktopus import UniformPrior
    >>> unif = UniformPrior(0., 1.)
    >>> unif(.5)
    -0.0
    >>> unif(1)
    inf
    """

    def __init__(self, lb, ub, name=None):
        self.lb = np.asarray([lb]).reshape(-1)
        self.ub = np.asarray([ub]).reshape(-1)
        if (self.lb >= self.ub).any():
            raise ValueError("The lower bounds should be smaller than the upper bounds.")
        self.name = name

    def __repr__(self):
        return "<UniformPrior(lb={}, ub={})>".format(self.lb, self.ub)

    @property
    def mean(self):
        """Returns the mean of the uniform distributions
        """
        return 0.5 * (self.lb + self.ub)

    @property
    def variance(self):
        """Returns the variance of the uniform distributions
        """
        return (self.ub - self.lb) ** 2 / 12.

    def evaluate(self, params):
        if (self.lb <= params).all() and (params < self.ub).all():
            return - np.log(1. / (self.ub - self.lb)).sum()
        return np.inf

    def gradient(self, params):
        if (self.lb <= params).all() and (params < self.ub).all():
            return 0.0
        return np.inf


class GaussianPrior(Prior):
    """Computes the negative log pdf for a n-dimensional independent Gaussian
    distribution.

    Attributes
    ----------
    mean : scalar or array-like
        Mean
    var : scalar or array-like
        Variance

    Examples
    --------
    >>> from oktopus import GaussianPrior
    >>> prior = GaussianPrior(0, 1)
    >>> prior(2.)
    2.0
    """

    def __init__(self, mean, var, name=None):
        self.mean = np.asarray([mean]).reshape(-1)
        self.var = np.asarray([var]).reshape(-1)
        self.name = name

    def __repr__(self):
        return "<GaussianPrior(mean={}, var={})>".format(self.mean, self.var)

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

    def gradient(self, params):
        return ((params - self.mean) / self.var).sum()

class LaplacianPrior(Prior):
    """Computes the negative log pdf for a n-dimensional independent Laplacian
    random variable.

    Attributes
    ----------
    mean : scalar or array-like
        Mean
    var : scalar or array-like
        Variance

    Examples
    --------
    >>> from oktopus import LaplacianPrior
    >>> prior = LaplacianPrior(0, 2)
    >>> prior(1.)
    1.0
    """

    def __init__(self, mean, var, name=None):
        self.mean = np.asarray([mean]).reshape(-1)
        self.var = np.asarray([var]).reshape(-1)
        self.name = name

    def __repr__(self):
        return "<LaplacianPrior(mean={}, var={})>".format(self.mean, self.var)

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
        return (abs(params - self.mean) / np.sqrt(.5 * self.var)).sum()
