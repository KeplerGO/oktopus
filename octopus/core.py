from abc import ABC, abstractmethod
from scipy.optimize import minimize


__all__ = ['LossFunction']


class LossFunction(ABC):
    """An abstract class for an arbitrary loss (cost) function.
    This type of function appears frequently in estimation problems where
    the best estimator (given a set of observed data) is the one which
    minimizes some sort of objective function.
    """

    @abstractmethod
    def evaluate(self, params):
        """
        Returns the loss function evaluated at params.
        """
        pass

    def __call__(self, params):
        return self.evaluate(params)

    def fit(self, x0, method='Nelder-Mead', **kwargs):
        """
        Minimizes the loss function using `scipy.optimize.minimize`.

        Parameters
        ----------
        x0 : ndarray
            Initial guesses on the parameter estimates
        method : str
            Optimization algorithm
        kwargs : dict
            Dictionary for additional arguments. See scipy.optimize.minimize

        Return
        ------
        opt_result : scipy.optimize.OptimizeResult object
            Object containing the results of the optimization process.
            Note: this is also stored in self.opt_result.
        """
        self.opt_result = minimize(self.evaluate, x0=x0, method=method,
                                   **kwargs)
        return self.opt_result
