import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

__all__ = ['MultinomialLikelihood']

class MultinomialLikelihood(object):
    """
    Parameters
    ----------
    data : ndarray
        Observed data.
    prob_model : callable
        Events probabilities of the multinomial distribution.
    """

    def __init__(self, data, prob_model):
        self.data = data
        self.prob_model = prob_model

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
        """
        pass
