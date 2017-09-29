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

        Parameters
        ----------
        params : ndarray
            parameter vector of the model

        Returns
        -------
        loss_fun : scalar
            Returns the scalar value of the loss function evaluated at
            **params**.
        """
        pass

    def __call__(self, params):
        """Calls :func:`evaluate`."""
        return self.evaluate(params)

    def fit(self, x0, method='Nelder-Mead', **kwargs):
        """
        Minimizes the :func:`evaluate` function using :func:`scipy.optimize.minimize`.

        Parameters
        ----------
        x0 : ndarray
            Initial guesses on the parameter estimates
        method : str
            Optimization algorithm
        kwargs : dict
            Dictionary for additional arguments. See :func:`scipy.optimize.minimize`

        Returns
        -------
        opt_result : :class:`scipy.optimize.OptimizeResult` object
            Object containing the results of the optimization process.
            Note: this is also stored in **self.opt_result**.
        """
        self.opt_result = minimize(self.evaluate, x0=x0, method=method,
                                   **kwargs)
        return self.opt_result
