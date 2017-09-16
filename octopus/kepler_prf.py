# Let's borrow as much as possible from the photutils framework
# However, this should me more generalizable, i.e., should accept
# priors on the parameters by allowing different loss (objective) functions
# remember: a prior is just a regularization term on the objective function
# User should be free to either optimize or do MCMC
# begin with the simplest example: fitting Kepler's PRF to a TPF
# generate movie of the residuals

from .models import get_initial_guesses
from .core import PoissonLikelihood
from abc import ABC, abstractmethod
import pandas as pd

class PRFPhotometry(ABC):
    # Let's restrict this for TPFs for now. Should be easily extensible though.
    @abstract_method
    def do_photometry(self, tpf, initial_guesses=None):
        pass

    @abstract_method
    def generate_residuals_movie(self):
        pass

class KeplerPRFPhotometry(PRFPhotometry):

    def __init__(self, prf_model, loss_function=PoissonLikelihood):
        self.prf_model = prf_model
        self.loss_function = loss_function

    def do_photometry(self, tpf, initial_guesses=None):
        if initial_guesses is None:
            # this must be clever enough to find the number of stars
            # great way to go is to use photutils.detection.DAOStarFinder
            initial_guesses, _ = get_inital_guesses(tpf.flux)

        results = []
        for t in range(len(tpf.time)):
            results = append(results,
                             self.loss_function(tpf.flux, self.prf_model).fit(initial_guesses).x)

        return np.array(results).reshape((tpf.shape[0], len(initial_guesses)))

    def generate_residuals_movie(self):
        pass
