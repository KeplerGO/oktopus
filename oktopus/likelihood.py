from abc import abstractmethod
import autograd.numpy as np
from autograd import jacobian
from .core import LossFunction


__all__ = ['Likelihood', 'MultinomialLikelihood', 'PoissonLikelihood',
           'GaussianLikelihood', 'MultivariateGaussianLikelihood']


class Likelihood(LossFunction):
    """
    Defines a Likelihood function.
    """

    def fisher_information_matrix(self):
        """
        Computes the Fisher Information Matrix using autograd

        Returns
        -------
        fisher : ndarray
            Fisher Information Matrix
        """
        n_params = len(self.opt_result.x)
        fisher = np.empty(shape=(n_params, n_params))
        grad_mean = []
        opt_params = self.opt_result.x

        for i in range(n_params):
            grad_mean.append(jacobian(self.mean, argnum=i))
        for i in range(n_params):
            for j in range(i, n_params):
                fisher[i, j] = ((grad_mean[i](*opt_params)
                                 * grad_mean[j](*opt_params)
                                 / self.mean(*opt_params)).sum())
                fisher[j, i] = fisher[i, j]
        return fisher

    def uncertainties(self):
        """
        Returns the uncertainties on the model parameters as the
        square root of the diagonal of the inverse of the Fisher
        Information Matrix.

        Returns
        -------
        inv_fisher : square root of the diagonal of the inverse of the Fisher
        Information Matrix
        """
        inv_fisher = np.linalg.inv(self.fisher_information_matrix())
        return np.sqrt(np.diag(inv_fisher))

    @abstractmethod
    def evaluate(self, params):
        """
        Evaluates the negative of the log likelihood function.

        Parameters
        ----------
        params : ndarray
            parameter vector of the model

        Returns
        -------
        neg_loglikelihood : scalar
            Returns the negative log likelihood function evaluated at
            ``params``.
        """
        pass


class MultinomialLikelihood(Likelihood):
    """
    Implements the negative log likelihood function for the Multinomial
    distribution. This class also contains a method to compute maximum
    likelihood estimators for the probabilities of the Multinomial
    distribution.

    Attributes
    ----------
    data : ndarray
        Observed count data.
    mean : callable
        Events probabilities of the multinomial distribution.

    Examples
    --------
    Suppose our data is divided in two classes and we would like to estimate
    the probability of occurence of each class with the condition that
    P(class_1) = 1 - P(class_2) = p. Suppose we have a sample with n_1 counts
    from class_1 and n_2 counts from class_2. Since the distribution of the
    number of counts is a binomial distribution, the MLE for P(class_1) is
    given as P(class_1) = n_1 / (n_1 + n_2), where n_i is the number of counts
    for class_i. The Fisher Information Matrix is given by
    F(n, p) = n / (p * (1 - p)). Let's see how we can estimate p.

    >>> from oktopus import MultinomialLikelihood
    >>> import autograd.numpy as np
    >>> counts = np.array([20, 30])
    >>> def ber_pmf(p):
    ...     return np.array([p, 1 - p])
    >>> logL = MultinomialLikelihood(data=counts, mean=ber_pmf)
    >>> p0 = 0.5 # our initial guess
    >>> p_hat = logL.fit(x0=p0)
    >>> p_hat.x
    array([ 0.4])
    >>> p_hat_unc = logL.uncertainties()
    >>> p_hat_unc
    array([ 0.06928203])
    >>> 20 / (20 + 30) # theorectical MLE
    0.4
    >>> np.sqrt(0.4 * 0.6 / (20 + 30)) # theorectical uncertanity
    0.069282032302755092
    """

    def __init__(self, data, mean):
        self.data = data
        self.mean = mean

    @property
    def n_counts(self):
        """
        Returns the sum of the number of counts in every bin.
        """
        return self.data.sum()

    def evaluate(self, params):
        return - (self.data * np.log(self.mean(*params))).sum()

    def fisher_information_matrix(self):
        return self.n_counts * super(MultinomialLikelihood,
                                     self).fisher_information_matrix()


class PoissonLikelihood(Likelihood):
    """
    Implements the negative log likelihood function for independent
    (possibly non-identically) distributed Poisson measurements.
    This class also contains a method to compute maximum likelihood estimators
    for the mean of the Poisson distribution.

    Attributes
    ----------
    data : ndarray
        Observed count data.
    mean : callable
        Mean of the Poisson distribution.
        Note: this model must be defined with autograd numpy wrapper.

    Examples
    --------
    Suppose we want to estimate the expected number of planes arriving at
    gate 50 terminal 1 at SFO airport in a given hour of a given day using
    some data.

    >>> import math
    >>> from oktopus import PoissonLikelihood
    >>> import numpy as np
    >>> import autograd.numpy as npa
    >>> np.random.seed(0)
    >>> toy_data = np.random.randint(1, 20, size=100)
    >>> def mean(l):
    ...     return npa.array([l])
    >>> logL = PoissonLikelihood(data=toy_data, mean=mean)
    >>> mean_hat = logL.fit(x0=10.5)
    >>> mean_hat.x
    array([ 9.28997498])
    >>> np.mean(toy_data) # theorectical MLE
    9.2899999999999991
    >>> mean_unc = logL.uncertainties()
    >>> mean_unc
    array([ 3.04794603])
    >>> math.sqrt(np.mean(toy_data)) # theorectical Fisher information
    3.047950130825634
    """

    def __init__(self, data, mean):
        self.data = data
        self.mean = mean

    def evaluate(self, params):
        return np.nansum(self.mean(*params) - self.data * np.log(self.mean(*params)))


class GaussianLikelihood(Likelihood):
    """
    Implements the likelihood function for independent
    (possibly non-identically) distributed Gaussian measurements
    with known variance.

    Attributes
    ----------
    data : ndarray
        Observed data
    mean : callable
        Mean model
    var : float or array-like
        Uncertainties on the observed data

    Examples
    --------
    The following example demonstrates how one can fit a maximum likelihood
    line to some data:

    >>> from oktopus import GaussianLikelihood
    >>> import autograd.numpy as np
    >>> from matplotlib import pyplot as plt
    >>> x = np.linspace(0, 10, 200)
    >>> np.random.seed(0)
    >>> fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    >>> def line(x, alpha, beta):
    ...     return alpha * x + beta
    >>> my_line = lambda a, b: line(x, a, b)
    >>> logL = GaussianLikelihood(fake_data, my_line, 4)
    >>> p0 = (1, 1) # dumb initial_guess for alpha and beta
    >>> p_hat = logL.fit(x0=p0)
    >>> p_hat.x # fitted parameters
    array([  2.96263393,  10.32860717])
    >>> p_hat_unc = logL.uncertainties() # get uncertainties on fitted parameters
    >>> p_hat_unc
    array([ 0.11568693,  0.55871623])
    >>> #plt.plot(x, fake_data, 'o')
    >>> #plt.plot(x, line(*p_hat.x))
    >>> # The exact values from linear algebra would be:
    >>> M = np.array([[np.sum(x * x), np.sum(x)], [np.sum(x), len(x)]])
    >>> alpha, beta = np.dot(np.linalg.inv(M), np.array([np.sum(fake_data * x), np.sum(fake_data)]))
    >>> alpha
    2.9626408752841442
    >>> beta
    10.328616609861584
    """

    def __init__(self, data, mean, var):
        self.data = data
        self.mean = mean
        self.var = var

    def evaluate(self, params):
        return np.nansum((self.data - self.mean(*params)) ** 2 / (2 * self.var))


class MultivariateGaussianLikelihood(Likelihood):
    """
    Implements the likelihood function of a multivariate gaussian distribution.

    Parameters
    ----------
    data : ndarray
        Observed data.
    mean : callable
        Mean model.
    cov : callable
        Kernel for the covariance matrix.
    dim : int
        Dimension (number of parameters) of the mean model.
    """

    def __init__(self, data, mean, cov, dim):
        self.data = data
        self.mean = mean
        self.cov = cov
        self.dim = dim

    def evaluate(self, params):
        """
        Computes the negative of the log likelihood function.

        Parameters
        ----------
        params : ndarray
            parameter vector of the mean model and covariance matrix
        """

        theta = params[:self.dim] # mean model parameters
        alpha = params[self.dim:] # kernel parameters (hyperparameters)

        residual = self.data - self.mean(*theta)

        return (np.linalg.slogdet(self.cov(*alpha))[1]
                + np.dot(residual.T, np.linalg.solve(self.cov(*alpha), residual)))

    def sample(self, **kwargs):
        return np.random.multivariate_normal(self.mean(*self.opt_result.x[:self.dim]),
                                             self.cov(*self.opt_result.x[self.dim:]),
                                             **kwargs)

    def fisher_information_matrix(self):
        raise NotImplementedError

    def uncertainties(self):
        raise NotImplementedError
