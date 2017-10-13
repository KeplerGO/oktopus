import autograd.numpy as np
from .loss import LossFunction
from .likelihood import PoissonLikelihood, GaussianLikelihood, MultivariateGaussianLikelihood


__all__ = ['Posterior', 'GaussianPosterior', 'PoissonPosterior', 'MultivariateGaussianPosterior']


class Posterior(LossFunction):
    """Defines a posterior distribution.

    Attributes
    ----------
    likelihood : callable or instance of :class:oktopus.Likelihood
        If callable, must provide a method called `evaluate` which returns
        the negative of the log likelihood.
    prior : callable or instance of :class:oktopus.Prior
        If callable, must provide a method called `evaluate` which returns
        the negative of the log of the distribution.

    Examples
    --------
    >>> import math
    >>> from oktopus import PoissonPosterior, PoissonLikelihood
    >>> import autograd.numpy as np
    >>> np.random.seed(0)
    >>> toy_data = np.random.randint(1, 20, size=100)
    >>> def mean(l):
    ...     return np.array([l])
    >>> logL = PoissonLikelihood(data=toy_data, mean=mean)
    >>> logP = Posterior(likelihood=logL, prior=logL.jeffreys_prior)
    >>> mean_hat = logP.fit(x0=10.5)
    >>> mean_hat.x
    array([ 9.2849762])
    >>> print(np.mean(toy_data)) # MLE estimate
    9.29
    """

    def __init__(self, likelihood, prior):
        self.loglikelihood = likelihood
        self.logprior = prior

    def evaluate(self, params):
        """Evaluates the negative of the log of the posterior at params.

        Parameters
        ----------
        params : scalar or array-like
            Value at which the posterior will be evaluated

        Returns
        -------
        value : scalar
            Value of the negative of the log of the posterior at params
        """
        return self.loglikelihood(params) + self.logprior(params)


class GaussianPosterior(Posterior):
    """
    Implements the negative log posterior distribution for uncorrelated
    (possibly non identically) distributed Gaussian measurements with known
    variances.

    Attributes
    ----------
    data : ndarray
        Observed data
    mean : callable
        Mean model
    var : scalar or array-like
        Uncertainties on the observed data.
    prior : callable
        Negative log prior as a function of the parameters
        See UniformPrior

    Examples
    --------
    >>> from oktopus import GaussianPosterior, GaussianPrior, UniformPrior, JointPrior
    >>> import autograd.numpy as np
    >>> #from matplotlib import pyplot as plt
    >>> x = np.linspace(0, 10, 200)
    >>> np.random.seed(0)
    >>> fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    >>> def line(x, slope, intercept):
    ...     return slope * x + intercept
    >>> my_line = lambda slope, intercept: line(x, slope, intercept)
    >>> slope_prior = UniformPrior(lb=1, ub=10)
    >>> intercept_prior = UniformPrior(lb=5, ub=20)
    >>> joint_prior = JointPrior(slope_prior, intercept_prior)
    >>> logP = GaussianPosterior(data=fake_data, mean=my_line, var=4, prior=joint_prior)
    >>> p0 = (slope_prior.mean, intercept_prior.mean) # initial guesses for slope and intercept
    >>> p_hat = logP.fit(x0=p0)
    >>> p_hat.x # fitted parameters
    array([  2.9626486 ,  10.32858499])
    >>> #plt.plot(x, fake_data, 'o')
    >>> #plt.plot(x, line(*p_hat.x))
    >>> # The exact values from linear algebra are:
    >>> M = np.array([[np.sum(x * x), np.sum(x)], [np.sum(x), len(x)]])
    >>> slope, intercept = np.dot(np.linalg.inv(M), np.array([np.sum(fake_data * x), np.sum(fake_data)]))
    >>> print(slope)
    2.96264087528
    >>> print(intercept)
    10.3286166099
    """

    def __init__(self, data, mean, var, prior):
        self.data = data
        self.mean = mean
        self.logprior = prior
        self.loglikelihood = GaussianLikelihood(data, mean, var)


class PoissonPosterior(Posterior):
    """
    Implements the negative of the log posterior distribution for independent
    (possibly non-identically) distributed Poisson measurements.

    Attributes
    ----------
    data : ndarray
        Observed count data
    mean : callable
        Mean of the Poisson distribution
        Note: If you would like to get uncertainties by using the
        `uncertainties` method, then this model must be defined with autograd
        numpy wrapper
    prior : callable
        Negative log prior as a function of the parameters.
        See UniformPrior

    Examples
    --------
    >>> import math
    >>> from oktopus import PoissonPosterior, UniformPrior, GaussianPrior
    >>> import autograd.numpy as np
    >>> np.random.seed(0)
    >>> toy_data = np.random.randint(1, 20, size=100)
    >>> def mean(l):
    ...     return np.array([l])
    >>> logP = PoissonPosterior(data=toy_data, mean=mean, prior=UniformPrior(lb=1, ub=20))
    >>> mean_hat = logP.fit(x0=10.5)
    >>> mean_hat.x # MAP is the same of MLE for uniform prior
    array([ 9.28997498])
    >>> logP = PoissonPosterior(data=toy_data, mean=mean, prior=GaussianPrior(mean=10, var=4))
    >>> mean_hat = logP.fit(x0=10.5)
    >>> mean_hat.x
    array([ 9.30612488])
    """

    def __init__(self, data, mean, prior):
        self.data = data
        self.mean = mean
        self.logprior = prior
        self.loglikelihood = PoissonLikelihood(data, mean)


class MultivariateGaussianPosterior(Posterior):
    """
    Implements the posterior distribution for a multivariate gaussian distribution.

    Attributes
    ----------
    data : ndarray
        Observed data
    mean : callable
        Mean model
    cov : callable
        Kernel for the covariance matrix
    dim : int
        Dimension (number of parameters) of the mean model
    prior : callable
        Negative log prior as a function of the parameters.
        See :class:UniformPrior
    """

    def __init__(self, data, mean, cov, dim, prior):
        self.data = data
        self.mean = mean
        self.cov = cov
        self.dim = dim
        self.logprior = prior
        self.loglikelihood = MultivariateGaussianLikelihood(data, mean, cov, dim)
