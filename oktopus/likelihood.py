import sys
from abc import abstractmethod
import autograd.numpy as np
from autograd import jacobian
from .loss import LossFunction


__all__ = ['Likelihood', 'MultinomialLikelihood', 'PoissonLikelihood',
           'GaussianLikelihood', 'LaplacianLikelihood',
           'MultivariateGaussianLikelihood', 'BernoulliLikelihood',
           'BernoulliGaussianMixtureLikelihood']


class Likelihood(LossFunction):
    """
    Defines a Likelihood function.
    """

    def fisher_information_matrix(self, params):
        """
        Computes the Fisher Information Matrix.

        Returns
        -------
        fisher : ndarray
            Fisher Information Matrix
        """
        pass

    def uncertainties(self, params):
        """
        Returns the uncertainties on the model parameters as the
        square root of the diagonal of the inverse of the Fisher
        Information Matrix evaluated at ``params``.

        Returns
        -------
        inv_fisher : square root of the diagonal of the inverse of the Fisher
        Information Matrix
        """
        inv_fisher = np.linalg.inv(self.fisher_information_matrix(params))
        return np.sqrt(np.diag(inv_fisher))

    def jeffreys_prior(self, params):
        """
        Computes the negative of the log of Jeffrey's prior and evaluates it at ``params``.
        """
        return - 0.5 * np.linalg.slogdet(self.fisher_information_matrix(params))[1]

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
    r"""
    Implements the negative log likelihood function for the Multinomial
    distribution. This class also contains a method to compute maximum
    likelihood estimators for the probabilities of the Multinomial
    distribution.

    .. math::

        \arg \min_{\theta \in \Theta} - \sum_k y_k \cdot \log p_k(\theta)

    Attributes
    ----------
    data : ndarray
        Observed count data
    mean : callable
        Events probabilities of the multinomial distribution

    Examples
    --------
    Suppose our data is divided in two classes and we would like to estimate
    the probability of occurence of each class with the condition that
    :math:`P(class_1) = 1 - P(class_2) = p`. Suppose we have a sample with
    :math:`n_1` counts from :math:`class_1` and :math:`n_2` counts from
    :math:`class_2`. Assuming the distribution of the number of counts is a
    binomial distribution, the MLE for :math:`P(class_1)` is given as
    :math:`P(class_1) = \dfrac{n_1}{n_1 + n_2}`. The Fisher Information Matrix
    is given by :math:`F(n, p) = \dfrac{n}{p * (1 - p)}`. Let's see how we can
    estimate :math:`p`.

    >>> from oktopus import MultinomialLikelihood
    >>> import autograd.numpy as np
    >>> counts = np.array([20, 30])
    >>> def ber_pmf(p):
    ...     return np.array([p, 1 - p])
    >>> logL = MultinomialLikelihood(data=counts, mean=ber_pmf)
    >>> p0 = 0.5 # our initial guess
    >>> p_hat = logL.fit(x0=p0, method='Nelder-Mead')
    >>> p_hat.x
    array([ 0.4])
    >>> p_hat_unc = logL.uncertainties(p_hat.x)
    >>> p_hat_unc
    array([ 0.06928203])
    >>> 20. / (20 + 30) # theorectical MLE
    0.4
    >>> print(np.sqrt(0.4 * 0.6 / (20 + 30))) # theorectical uncertainty
    0.0692820323028
    """

    def __init__(self, data, mean):
        self.data = data
        self.mean = mean

    def __repr__(self):
        return "<MultinomialLikelihood(mean={})>".format(self.mean)

    @property
    def n_counts(self):
        """
        Returns the sum of the number of counts over all bin.
        """
        return np.nansum(self.data)

    def evaluate(self, params):
        return - np.nansum(self.data * np.log(self.mean(*params)))

    def fisher_information_matrix(self, params):
        n_params = len(np.atleast_1d(params))
        fisher = np.empty(shape=(n_params, n_params))

        if not hasattr(self.mean, 'gradient'):
            _grad = lambda mean, argnum, params: jacobian(mean, argnum=argnum)(*params)
        else:
            _grad = lambda mean, argnum, params: mean.gradient(*params)[argnum]

        grad_mean = [_grad(self.mean, i, params) for i in range(n_params)]
        mean = self.mean(*params)

        for i in range(n_params):
            for j in range(i, n_params):
                fisher[i, j] = np.nansum(grad_mean[i] * grad_mean[j] / mean)
                fisher[j, i] = fisher[i, j]

        return self.n_counts * fisher

    def gradient(self, params):
        # use the gradient if the model provides it.
        # if not, compute it using autograd.
        if not hasattr(self.mean, 'gradient'):
            _grad = lambda mean, argnum, params: jacobian(mean, argnum)(*params)
        else:
            _grad = lambda mean, argnum, params: mean.gradient(*params)[argnum]
        n_params = len(np.atleast_1d(params))
        grad_likelihood = np.array([])
        for i in range(n_params):
            grad = _grad(self.mean, i, params)
            grad_likelihood = np.append(grad_likelihood,
                                        - np.nansum(self.data * grad / self.mean(*params)))
        return grad_likelihood


class PoissonLikelihood(Likelihood):
    r"""
    Implements the negative log likelihood function for independent
    (possibly non-identically) distributed Poisson measurements.
    This class also contains a method to compute maximum likelihood estimators (MLE)
    for the mean of the Poisson distribution.

    More precisely, the MLE is computed as:

    .. math::

         \arg \min_{\theta \in \Theta} \sum_k \lambda_k(\theta) - y_k \cdot \log \lambda_k(\theta)

    Attributes
    ----------
    data : ndarray
        Observed count data
    mean : callable
        Mean of the Poisson distribution
        Note: If you want to compute uncertainties, this model must be defined
        with autograd numpy wrapper

    Notes
    -----
    See `here <https://mirca.github.io/geerts-conjecture/>`_ for the mathematical
    derivation of the Poisson likelihood expression.

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
    array([ 9.29000013])
    >>> print(np.mean(toy_data)) # theorectical MLE
    9.29
    >>> mean_unc = logL.uncertainties(mean_hat.x)
    >>> mean_unc
    array([ 3.04795015])
    >>> print("{:.5f}".format(math.sqrt(np.mean(toy_data)))) # theorectical Fisher information
    3.04795
    """

    def __init__(self, data, mean):
        self.data = data
        self.mean = mean

    def __repr__(self):
        return "<PoissonLikelihood(mean={})>".format(self.mean)

    def evaluate(self, params):
        return np.nansum(self.mean(*params) - self.data * np.log(self.mean(*params)))

    def fisher_information_matrix(self, params):
        n_params = len(np.atleast_1d(params))
        fisher = np.empty(shape=(n_params, n_params))

        if not hasattr(self.mean, 'gradient'):
            _grad = lambda mean, argnum, params: jacobian(mean, argnum=argnum)(*params)
        else:
            _grad = lambda mean, argnum, params: mean.gradient(*params)[argnum]

        grad_mean = [_grad(self.mean, i, params) for i in range(n_params)]
        mean = self.mean(*params)

        for i in range(n_params):
            for j in range(i, n_params):
                fisher[i, j] = np.nansum(grad_mean[i] * grad_mean[j] / mean)
                fisher[j, i] = fisher[i, j]

        return fisher

    def gradient(self, params):
        # use the gradient if the model provides it.
        # if not, compute it using autograd.
        if not hasattr(self.mean, 'gradient'):
            _grad = lambda mean, argnum, params: jacobian(mean, argnum)(*params)
        else:
            _grad = lambda mean, argnum, params: mean.gradient(*params)[argnum]
        n_params = len(np.atleast_1d(params))
        grad_likelihood = np.array([])
        for i in range(n_params):
            grad = _grad(self.mean, i, params)
            grad_likelihood = np.append(grad_likelihood,
                                        np.nansum(grad * (1 - self.data / self.mean(*params))))
        return grad_likelihood


class GaussianLikelihood(Likelihood):
    r"""
    Implements the likelihood function for independent
    (possibly non-identically) distributed Gaussian measurements
    with known variance.

    The maximum likelihood estimator is computed as:

    .. math::

         \arg \min_{\theta \in \Theta} \dfrac{1}{2}\sum_k \left(\dfrac{y_k - \mu_k(\theta)}{\sigma_k}\right)^2

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
    >>> from matplotlib import pyplot as plt # doctest: +SKIP
    >>> x = np.linspace(0, 10, 200)
    >>> np.random.seed(0)
    >>> fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    >>> def line(x, alpha, beta):
    ...     return alpha * x + beta
    >>> my_line = lambda a, b: line(x, a, b)
    >>> logL = GaussianLikelihood(fake_data, my_line, 4)
    >>> p0 = (1, 1) # dumb initial_guess for alpha and beta
    >>> p_hat = logL.fit(x0=p0, method='Nelder-Mead')
    >>> p_hat.x # fitted parameters
    array([  2.96263393,  10.32860717])
    >>> p_hat_unc = logL.uncertainties(p_hat.x) # get uncertainties on fitted parameters
    >>> p_hat_unc
    array([ 0.04874546,  0.28178535])
    >>> plt.plot(x, fake_data, 'o') # doctest: +SKIP
    >>> plt.plot(x, line(*p_hat.x)) # doctest: +SKIP
    >>> # The exact values from linear algebra would be:
    >>> M = np.array([[np.sum(x * x), np.sum(x)], [np.sum(x), len(x)]])
    >>> alpha, beta = np.dot(np.linalg.inv(M), np.array([np.sum(fake_data * x), np.sum(fake_data)]))
    >>> print(alpha)
    2.96264087528
    >>> print(beta)
    10.3286166099
    """

    def __init__(self, data, mean, var):
        self.data = data
        self.mean = mean
        self.var = var

    def __repr__(self):
        return "<GaussianLikelihood(mean={}, var={})>".format(self.mean, self.var)

    def evaluate(self, params):
        r = self.data - self.mean(*params)
        return 0.5 * np.nansum(r * r / self.var)

    def gradient(self, params):
        # use the gradient if the model provides it.
        # if not, compute it using autograd.
        if not hasattr(self.mean, 'gradient'):
            _grad = lambda mean, argnum, params: jacobian(mean, argnum)(*params)
        else:
            _grad = lambda mean, argnum, params: mean.gradient(*params)[argnum]
        n_params = len(np.atleast_1d(params))
        grad_likelihood = np.array([])
        r = self.data - self.mean(*params)
        for i in range(n_params):
            grad = _grad(self.mean, i, params)
            grad_likelihood = np.append(grad_likelihood,
                                        -np.nansum(r * grad / self.var))
        return grad_likelihood

    def fisher_information_matrix(self, params):
        """
        Computes the Fisher Information Matrix.

        Returns
        -------
        fisher : ndarray
            Fisher Information Matrix
        """
        n_params = len(np.atleast_1d(params))
        fisher = np.zeros(shape=(n_params, n_params))

        if not hasattr(self.mean, 'gradient'):
            _grad = lambda mean, argnum, params: jacobian(mean, argnum=argnum)(*params)
        else:
            _grad = lambda mean, argnum, params: mean.gradient(*params)[argnum]

        grad_mean = [_grad(self.mean, i, params) for i in range(n_params)]

        for i in range(n_params):
            for j in range(i, n_params):
                fisher[i, j] = np.nansum(grad_mean[i] * grad_mean[j])
                fisher[j, i] = fisher[i, j]

        return fisher / self.var


class LaplacianLikelihood(Likelihood):
    r"""
    Implements the likelihood function for independent
    (possibly non-identically) distributed Laplacian measurements
    with known error bars.

    .. math::

         \arg \min_{\theta \in \Theta} \sum_k \dfrac{|y_k - \mu_k(\theta)|}{\sigma_k}

    Attributes
    ----------
    data : ndarray
        Observed data
    mean : callable
        Mean model
    var : float or array-like
        Uncertainties on the observed data
    """

    def __init__(self, data, mean, var):
        self.data = data
        self.mean = mean
        self.var = var

    def __repr__(self):
        return "<LaplacianLikelihood(mean={}, var={})>".format(self.mean, self.var)

    def evaluate(self, params):
        return np.nansum(np.abs(self.data - self.mean(*params)) / np.sqrt(.5 * self.var))

    def fisher_information_matrix(self, params):
        raise NotImplementedError

    def uncertainties(self, params):
        raise NotImplementedError


class MultivariateGaussianLikelihood(Likelihood):
    """
    Implements the likelihood function of a multivariate gaussian distribution.

    Parameters
    ----------
    data : ndarray
        Observed data.
    mean : callable
        Mean model.
    cov : ndarray or callable
        If callable, the parameters of the covariance matrix
        will be fitted.
    dim : int
        Number of parameters of the mean model.
    """

    def __init__(self, data, mean, cov, dim):
        self.data = data
        self.mean = mean
        self.cov = cov
        self.dim = dim
        self._is_cov_callable = callable(self.cov)
        if not self._is_cov_callable:
            self._cov_inv = np.linalg.inv(self.cov)

    def __repr__(self):
        return "<MultivariateGaussianLikelihood(mean={}, cov={})>".format(self.mean, self.cov)

    def evaluate(self, params):
        """
        Computes the negative of the log likelihood function.

        Parameters
        ----------
        params : ndarray
            parameter vector of the mean model and covariance matrix
        """
        if callable(self.cov):
            theta = params[:self.dim] # mean model parameters
            alpha = params[self.dim:] # kernel parameters (hyperparameters)
            mean = self.mean(*theta)
            cov = self.cov(*alpha)
            self._cov_inv = np.linalg.inv(cov)
        else:
            mean = self.mean(*params)
            cov = self.cov

        residual = self.data - mean

        return (np.linalg.slogdet(cov)[1]
                + np.nansum(residual * (self._cov_inv * residual.T)))

    def gradient(self, params):
        # use the gradient if the model provides it.
        # if not, compute it using autograd.
        if not hasattr(self.mean, 'gradient'):
            _grad = lambda mean, argnum, params: jacobian(mean, argnum)(*params)
        else:
            _grad = lambda mean, argnum, params: mean.gradient(*params)[argnum]
        n_params = len(np.atleast_1d(params))
        grad_likelihood = np.array([])
        r = self.data - self.mean(*params)
        for i in range(n_params):
            grad = _grad(self.mean, i, params)
            grad_likelihood = np.append(grad_likelihood,
                                        -np.nansum(grad * self._cov_inv * r))
        return grad_likelihood

    def fisher_information_matrix(self, params):
        n_params = len(np.atleast_1d(params))
        fisher = np.zeros(shape=(n_params, n_params))

        if not hasattr(self.mean, 'gradient'):
            _grad = lambda mean, argnum, params: jacobian(mean, argnum=argnum)(*params)
        else:
            _grad = lambda mean, argnum, params: mean.gradient(*params)[argnum]

        grad_mean = [_grad(self.mean, i, params) for i in range(n_params)]

        for i in range(n_params):
            for j in range(i, n_params):
                fisher[i, j] = np.nansum(grad_mean[i] * self._cov_inv * grad_mean[j])
                fisher[j, i] = fisher[i, j]

        return fisher


class BernoulliLikelihood(Likelihood):
    r"""Implements the negative log likelihood function for independent
    (possibly non-identical distributed) Bernoulli random variables.
    This class also contains a method to compute maximum likelihood estimators
    for the probability of a success.

    More precisely, the MLE is computed as

    .. math::
        \arg \min_{\theta \in \Theta} - \sum_{i=1}^{n} y_i\log\pi_i(\mathbf{\theta}) + (1 - y_i)\log(1 - \pi_i(\mathbf{\theta}))

    Attributes
    ----------
    data : array-like
        Observed data
    mean : callable
        A functional form that defines the model for the probability of success

    Examples
    --------
    >>> import numpy as np
    >>> from oktopus import BernoulliLikelihood, UniformPrior, Posterior
    >>> from oktopus.models import ConstantModel as constant
    >>> # generate integer fake data in the set {0, 1}
    >>> np.random.seed(0)
    >>> y = np.random.choice([0, 1], size=401)
    >>> # create a model
    >>> p = constant()
    >>> # perform optimization
    >>> ber = BernoulliLikelihood(data=y, mean=p)
    >>> unif = UniformPrior(lb=0., ub=1.)
    >>> pp = Posterior(likelihood=ber, prior=unif)
    >>> result = pp.fit(x0=.3, method='powell')
    >>> # get best fit parameters
    >>> print(np.round(result.x, 3))
    0.529
    >>> print(np.round(np.mean(y>0), 3)) # theorectical MLE
    0.529
    >>> # get uncertainties on the best fit parameters
    >>> print(ber.uncertainties([result.x]))
    [ 0.0249277]
    >>> # theorectical uncertainty
    >>> print(np.sqrt(0.528678304239 * (1 - 0.528678304239) / 401))
    0.0249277036876
    """

    def __init__(self, data, mean):
        self.data = np.asarray(data)
        self.mean = mean

    def evaluate(self, theta):
        mean_theta = self.mean(*theta)
        return - np.nansum(self.data * np.log(mean_theta)
                           + (1. - self.data) * np.log(1. - mean_theta))

    def gradient(self, theta):
        mean_theta = self.mean(*theta)
        grad = self.mean.gradient(*theta)
        return - np.nansum(self.data * grad / mean_theta
                           - (1 - self.data) * grad / (1 - mean_theta),
                           axis=-1)

    def fisher_information_matrix(self, theta):
        n_params = len(np.atleast_1d(theta))
        fisher = np.empty(shape=(n_params, n_params))
        grad_mean = self.mean.gradient(*theta)
        mean = self.mean(*theta)

        for i in range(n_params):
            for j in range(i, n_params):
                fisher[i, j] = (grad_mean[i] * grad_mean[j] / mean).sum()
                fisher[j, i] = fisher[i, j]
        return len(self.data) * fisher / (1 - self.mean(*theta))

    def uncertainties(self, theta):
        inv_fisher = np.linalg.inv(self.fisher_information_matrix(theta))
        return np.sqrt(np.diag(inv_fisher))


class BernoulliGaussianMixtureLikelihood(Likelihood):
    r"""Implements the likelihood of :math:`Z_i = X_i + Y_i`,
    such that :math:`X_i` is a Bernoulli random variable, with
    probability of success :math:`p_i`, and :math:`Y_i` is a
    normal random variable with zero mean and known variance :math:`sigma^2`.

    The loglikelihood is given as

    .. math::

             \log p(z^n) = \sum_{i=1}^{n} \log \left\[(1 - p_i)  \mathcal{N}(0,
             \sigma^2) + p_i \mathcal{N}(1, \sigma^2)\right\]

    Examples
    --------
    >>> import numpy as np
    >>> from oktopus import BernoulliGaussianMixtureLikelihood, UniformPrior, Posterior
    >>> from oktopus.models import ConstantModel as constant
    >>> # generate integer fake data in the set {0, 1}
    >>> np.random.seed(0)
    >>> y = np.append(np.random.normal(size=30), 1+np.random.normal(size=70))
    >>> # create a model
    >>> p = constant()
    >>> # build likelihood
    >>> ll = BernoulliGaussianMixtureLikelihood(data=y, mean=p, var=1.)
    >>> unif = UniformPrior(lb=0., ub=1.)
    >>> pp = Posterior(likelihood=ll, prior=unif)
    >>> result = pp.fit(x0=.3, method='powell')
    >>> # get best fit parameters
    >>> print(np.round(result.x, 3))
    0.803
    >>> print(np.mean(y>0)) # theorectical MLE
    0.78
    """

    def __init__(self, data, mean, var=1.):
        self.data = np.asarray(data)
        self.mean = mean
        self.var = var

    def evaluate(self, theta):
        p = self.mean(*theta)
        N0 = np.exp(-self.data ** 2 / (2 * self.var))
        N1 = np.exp(-(self.data - 1) ** 2 / (2 * self.var))
        return -np.nansum(np.log((1 - p) * N0 + p * N1))
