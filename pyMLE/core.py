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
    prob_model : callable
        Events probabilities of the multinomial distribution.
    """

    def __init__(self, data, prob_model):
        self.data = data
        self.prob_model = prob_model

    @property
    def n_bins(self):
        return len(self.data)

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
        return - (self.data * np.log(self.prob_model(*params))).sum()

    def fit(self, x0, **kwargs):
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
        self.opt_result = minimize(self.evaluate, x0=x0, **kwargs)
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
        grad_prob_model = []
        opt_params = self.opt_result.x

        for i in range(n_params):
            grad_prob_model.append(jacobian(self.prob_model, argnum=i))

        for i in range(n_params):
            for j in range(i, n_params):
                fisher[i, j] = ((grad_prob_model[i](*opt_params) *
                                 grad_prob_model[j](*opt_params) /
                                 self.prob_model(*opt_params)).sum())
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
