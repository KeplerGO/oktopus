from abc import ABC, abstractmethod
import autograd.numpy as np
from autograd import jacobian
from scipy.optimize import minimize


__all__ = ['MultinomialLikelihood', 'PoissonLikelihood', 'GaussianLikelihood',
           'MultivariateGaussianLikelihood']


class Likelihood(ABC):
    @abstractmethod
    def evaluate(self, params):
        """
        Returns the negative of the log likelihood function.

        Parameters
        ----------
        params : ndarray
            parameter vector of the model
        """
        pass

    def fit(self, x0, method='Nelder-Mead', **kwargs):
        """
        Find the maximum likelihood estimator of the parameter vector by
        minimizing the negative of the log likelihood function.

        Parameters
        ----------
        x0 : ndarray
            Initial guesses on the parameter estimates
        kwargs : dict
            Dictionary for additional arguments. See scipy.optimize.minimize.

        Return
        ------
        opt_result : scipy.optimize.OptimizeResult object
            Object containing the results of the optimization process.
            Note: this is also store in self.opt_result.
        """
        self.opt_result = minimize(self.evaluate, x0=x0, method=method,
                                   **kwargs)
        return self.opt_result

    def fisher_information_matrix(self):
        """
        Computes the Fisher Information Matrix

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
                fisher[i, j] = ((grad_mean[i](*opt_params) *
                                 grad_mean[j](*opt_params) /
                                 self.mean(*opt_params)).sum())
                fisher[j, i] = fisher[i, j]
        return fisher

    def uncertainties(self):
        """
        Returns the uncertainties on the model parameters as the
        square root of the diagonal of the inverse of the Fisher
        Information Matrix.

        Returns
        -------
        unc : square root of the diagonal of the inverse of the Fisher
        Information Matrix
        """
        inv_fisher = np.linalg.inv(self.fisher_information_matrix())
        unc = np.sqrt(np.diag(inv_fisher))
        return unc

class MultinomialLikelihood(Likelihood):
    """
    Implements the likelihood function for the Multinomial distribution.
    This class also contains method to compute maximum likelihood estimators
    for the probabilities of the Multinomial distribution.

    Parameters
    ----------
    data : ndarray
        Observed count data.
    pmf : callable
        Events probabilities of the multinomial distribution.
        Note: this model must be defined with autograd numpy wrapper.

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

    >>> from pyMLE import MultinomialLikelihood
    >>> import autograd.numpy as np
    >>> counts = np.array([20, 30])
    >>> def ber_pmf(p):
            return np.array([p, 1 - p])
    >>> logL = MultinomialLikelihood(data=counts, pmf=ber_pmf)
    >>> p0 = 0.5 # our initial guess
    >>> p_hat = logL.fit(x0=p0)
    >>> p_hat.x
        array([0.4])
    >>> p_hat_unc = logL.uncertainties()
    >>> p_hat_unc
    >>> array([ 0.06928203])
    >>> 20 / (20 + 30) # theorectical MLE
        0.4
    >>> np.sqrt(0.4 * 0.6 / (20 + 30)) # theorectical uncertanity
        0.069282032302755092
    """

    def __init__(self, data, pmf):
        self.data = data
        self.pmf = pmf

    @property
    def n_counts(self):
        return self.data.sum()

    def evaluate(self, params):
        """
        Returns the negative of the log likelihood function.

        Parameters
        ----------
        params : ndarray
            parameter vector of the model
        """
        return - (self.data * np.log(self.pmf(*params))).sum()

    def fisher_information_matrix(self):
        """
        Computes the Fisher Information Matrix

        Returns
        -------
        fisher : ndarray
            Fisher Information Matrix
        """

        return self.n_counts * super(MultinomialLikelihood,
                                     self).fisher_information_matrix()


class PoissonLikelihood(Likelihood):
    """
    Implements the likelihood function for independent
    (possibly non-identically) distributed Poission measurements.
    This class also contains method to compute maximum likelihood estimators
    for the mean of the Poisson distribution.

    Parameters
    ----------
    data : ndarray
        Observed count data.
    mean : callable
        Mean of the Poisson distribution.
        Note: this model must be defined with autograd numpy wrapper.

    Examples
    --------

    """

    def __init__(self, data, mean):
        self.data = data
        self.mean = mean

    def evaluate(self, params):
        """
        Returns the negative of the log likelihood function.

        Parameters
        ----------
        params : ndarray
            parameter vector of the model
        """
        return  (self.mean(*params) - self.data * np.log(self.mean(*params))).sum()


class GaussianLikelihood(Likelihood):
    """
    Implements the likelihood function for independent
    (possibly non-identically) distributed Gaussian measurements
    with known variance.

    Examples
    --------
    The following example demonstrates how one can fit a maximum likelihood
    line to some data:

    >>> from pyMLE import GaussianLikelihood
    >>> import autograd.numpy as np
    >>> from matplotlib import pyplot as plt
    >>> x = np.linspace(0, 10, 200)
    >>> fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    >>> def line(x, alpha, beta):
    ...     return alpha * x + beta
    >>> my_line = lambda a, b: line(x, a, b)
    >>> logL = GaussianLikelihood(fake_data, my_line, 4)
    >>> p0 = (1, 1) # dumb initial_guess for alpha and beta
    >>> p_hat = logL.fit(x0=p0)
    >>> p_hat.x # fitted parameters
        array([ 3.00888507,  9.83031969])
    >>> p_hat_unc = logL.uncertainties() # get uncertainties on fitted parameters
    >>> p_hat_unc
        array([ 0.11466403,  0.55011405])
    >>> plt.plot(x, fake_data, 'o')
    >>> plt.plot(x, line(*p_hat.x))
    >>> # The exact values from linear algebra would be:
    >>> M = np.array([[np.sum(x * x), np.sum(x)], [np.sum(x), len(x)]])
    >>> alpha, beta = np.dot(np.inv(M), np.array([np.sum(fake_data * x), np.sum(fake_data)]))
    >>> alpha
        3.0088767708640294
    >>> beta
        9.8303661197685983
    """

    def __init__(self, data, mean, var):
        self.data = data
        self.mean = mean
        self.var = var

    def evaluate(self, params):
        """
        Returns the negative of the log likelihood function.

        Parameters
        ----------
        params : ndarray
            parameter vector of the model
        """
        return (1 / self.var * (self.data - self.mean(*params)) ** 2).sum()

class MultivariateGaussianLikelihood(Likelihood):

    def __init__(self, data, mean, cov, dim):
        self.data = data
        self.mean = mean
        self.cov = cov
        self.dim = dim # number of parameters of the mean model

    def evaluate(self, params):
        """
        Returns the negative of the log likelihood function.

        Parameters
        ----------
        params : ndarray
            parameter vector of the mean model and covariance matrix
        """

        theta = params[:self.dim] # mean model parameters
        alpha = params[self.dim:] # kernel parameters (hyperparameters)

        residual = self.data - self.mean(*theta)

        return (np.linalg.slogdet(self.cov(*alpha))[1]
                + np.dot(residual, np.linalg.solve(self.cov(*alpha), residual)))

    def fisher_information_matrix(self):
        raise NotImplementedError

    def uncertainteis(self):
        raise NotImplementedError
