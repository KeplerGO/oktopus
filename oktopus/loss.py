from abc import abstractmethod
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping


try:
    from skopt import gp_minimize
except ImportError:
    pass


__all__ = ['LossFunction', 'L1Norm']


class LossFunction(object):
    """An abstract class for an arbitrary loss (cost) function.
    This type of function appears frequently in estimation problems where
    the best estimator (given a set of observed data) is the one which
    minimizes some sort of objective function.
    """

    @abstractmethod
    def evaluate(self, params):
        """
        Returns the loss function evaluated at params

        Parameters
        ----------
        params : ndarray
            parameter vector of the model

        Returns
        -------
        loss_fun : scalar
            Returns the scalar value of the loss function evaluated at
            **params**
        """
        pass

    def __call__(self, params):
        """Calls :func:`evaluate`"""
        return self.evaluate(params)

    def fit(self, optimizer='minimize', **kwargs):
        """
        Minimizes the :func:`evaluate` function using :func:`scipy.optimize.minimize`,
        :func:`scipy.optimize.differential_evolution`,
        :func:`scipy.optimize.basinhopping`, or :func:`skopt.gp.gp_minimize`.

        Parameters
        ----------
        optimizer : str
            Optimization algorithm
        kwargs : dict
            Dictionary for additional arguments. See :func:`scipy.optimize.minimize`

        Returns
        -------
        opt_result : :class:`scipy.optimize.OptimizeResult` object
            Object containing the results of the optimization process.
            Note: this is also stored in **self.opt_result**.
        """

        if optimizer == 'minimize':
            self.opt_result = minimize(self.evaluate, **kwargs)
        elif optimizer == 'differential_evolution':
            self.opt_result = differential_evolution(self.evaluate, **kwargs)
        elif optimizer == 'basinhopping':
            self.opt_result = basinhopping(self.evaluate, **kwargs)
        elif optimizer == 'gp_minimize':
            self.opt_result = gp_minimize(self.evaluate, **kwargs)
        else:
            raise ValueError("method {} is not available".format(method))

        return self.opt_result

class L1Norm(LossFunction):
    r"""Defines the L1 Norm loss function. L1 norm is usually useful
    to optimize the "median" model, i.e., it is more robust to
    outliers than the quadratic loss function.

    .. math::

        \arg \min_{\theta \in \Theta} \sum_k |y_k - f(x_k, \theta)|

    Attributes
    ----------
    data : array-like
        Observed data
    model : callable
        A functional form that defines the model
    regularization : callable
        A functional form that defines the regularization term

    Examples
    --------
    >>> from oktopus import L1Norm
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> data = np.random.exponential(size=50)
    >>> def constant_model(a):
    ...     return a
    >>> l1norm = L1Norm(data=data, model=constant_model)
    >>> result = l1norm.fit(x0=np.mean(data))
    >>> result.x
    array([ 0.83998338])
    >>> print(np.median(data)) # the analytical solution
    0.839883776803
    """

    def __init__(self, data, model, regularization=None):
        self.data = data
        self.model = model
        self.regularization = regularization
        if self.regularization is None:
            self._evaluate = self._evaluate_wo_regularization
        else:
            self._evaluate = self._evaluate_w_regularization

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, func):
        if func is not None:
            self._regularization = func
            self._evaluate = self._evaluate_w_regularization
        else:
            self._regularization = None
            self._evaluate = self._evaluate_wo_regularization

    def _evaluate_wo_regularization(self, *params):
        return  np.nansum(np.absolute(self.data - self.model(*params)))

    def _evaluate_w_regularization(self, *params):
        return  np.nansum(np.absolute(self.data - self.model(*params[:-1]))
                          + params[-1] * self.regularization(*params[:-1]))

    def evaluate(self, params):
        return self._evaluate(*params)

