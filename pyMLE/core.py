import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import jacobian
from scipy.optimize import minimize, curve_fit


__all__ = ['MultinomialLikelihood']


class MultinomialLikelihood(object):
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
    from class_1 and n_2 counts from class_2. Since this constitutes a binomial
    distribution, the MLE for P(class_1) is given as
    P(class_1) = n_1 / (n_1 + n_2), where n_i is the number of counts for
    class_i. The Fisher Information Matrix is given by
    F(n, p) = n / (p * (1 - p)). Let's see how we can estimate p.

    >>> from pyMLE import MultinomialLikelihood
    >>> import autograd.numpy as np
    >>> counts = np.array([20, 30])
    >>> def bin_pmf(p):
            return np.array([p, 1 - p])
    >>> logL = MultinomialLikelihood(data=counts, pmf=bin_pmf)
    >>> p0 = 0.5 # our initial guess
    >>> p_hat = logL.fit(x0=p0)
    >>> p_hat
        array([0.4])
    >>> p_hat_unc = logL.uncertanties()
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
        grad_pmf = []
        opt_params = self.opt_result.x

        for i in range(n_params):
            grad_pmf.append(jacobian(self.pmf, argnum=i))

        for i in range(n_params):
            for j in range(i, n_params):
                fisher[i, j] = ((grad_pmf[i](*opt_params) *
                                 grad_pmf[j](*opt_params) /
                                 self.pmf(*opt_params)).sum())
                fisher[j, i] = fisher[i, j]
        fisher = self.n_counts * fisher
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
