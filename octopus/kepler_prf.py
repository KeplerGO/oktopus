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
import scipy
import pandas as pd
import math
from pyke import TargetPixelFile

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

class KeplerPRF(KeplerTargetPixelFile):
    """
    Kepler's Pixel Response Function

    This class provides the necessary interface to load the Kepler PSF
    calibration files and to create a model that can be fit as a function
    of flux and centroid position.
    """

    def __init__(self, file_dir):
        self.file_dir = file_dir

    def prf_to_detector(self, F, xo, yo):
        self.prf_model = np.zeros((np.size(self.y), np.size(self.x)))

        FRCx, INTx = math.modf(xo)
        FRCy, INTy = math.modf(yo)

        if FRCx > 0.5:
            FRCx = 1.0 - FRCx
            INTx = 1.0 + INTx

        if FRCy > 0.5:
            FRCy = 1.0 - FRCy
            INTy = 1.0 + INTy

        FRCx = 1.0 - FRCx
        FRCy = 1.0 - FRCy

        for (j, yj) in enumerate(self.y):
            for (i, xi) in enumerate(self.x):
                xx = xi - INTx + FRCx
                yy = yj - INTy + FRCy
                dx = xx
                dy = yy
                self.prf_model[j, i] = (self.prf_model[j, i]
                                        + F * self.interpolate(dy, dx))

        return self.prf_model

    def evaluate(self, F, xo, yo):
        return self.prf_to_detector(F, xo, yo):

    def interpolate(self, X, Y, values):
        # should return an instance of scipy.interpolate.RectBivariateSpline(X, Y, values)

    def read_prf_calibration_file(self, path, ext):
        prf_cal_file = pyfits.open(path)
        data = prf_cal_file[ext].data
        crpix1p = prf_cal_file[ext].header['CRPIX1P']
        crpix2p = prf_cal_file[ext].header['CRPIX2P']
        crval1p = prf_cal_file[ext].header['CRVAL1P']
        crval2p = prf_cal_file[ext].header['CRVAL2P']
        cdelt1p = prf_cal_file[ext].header['CDELT1P']
        cdelt2p = prf_cal_file[ext].header['CDELT2P']
        prf_cal_file.close()
        return data, crpix1p, crpix2p, crval1p, crval2p, cdelt1p, cdelt2p

    def prepare_prf(self):



#def read_and_interpolate_prf(prfdir, module, output, column, row, xdim, ydim,
#                             verbose=False, logfile='kepprf.log'):
#    """
#    Read PRF file and prepare the data to be used for evaluating PRF.
#
#    Parameters
#    ----------
#    prfdir : str
#        The full or relative directory path to a folder containing the Kepler
#        PSF calibration. Calibration files can be downloaded from the Kepler
#        focal plane characteristics page at the MAST.
#    module : str
#        The 'MODULE' keyword from TPF file.
#    output : str
#        The 'OUTPUT' keyword from TPF file.
#    column : int
#        The '1CRV5P' keyword from TPF[1] file.
#    row : int
#        The '2CRV5P' keyword from TPF[1] file.
#    xdim : int
#        The first part of the 'TDIM5' keyword from TPF[1] file.
#    ydim : int
#        The second part of the 'TDIM5' keyword from TPF[1] file.
#    verbose : boolean
#        Print informative messages and warnings to the shell and logfile?
#    logfile : string
#        Name of the logfile containing error and warning messages.
#
#    Returns
#    -------
#    splineInterpolation
#        You can get PRF at given position:
#        kepfunc.PRF2DET([flux], [x], [y], DATx, DATy, 1.0, 1.0, 0.0, splineInterpolation)
#    DATx : numpy.array
#        X-axis coordiantes of pixels for given TPF
#    DATy : numpy.array
#        Y-axis coordinates of pixels for given TPF
#    prf : numpy.array
#        PRF interpolated to given position on the camera
#    PRFx : numpy.array
#        X-axis coordinates of prf values
#    PRFy : numpy.array
#        Y-axis coordinates of prf values
#    PRFx0 : int
#    PRFy0 : int
#    cdelt1p : numpy.array
#        CDELT1P values from 5 HDUs of PRF file.
#    cdelt2p : numpy.array
#        CDELT2P values from 5 HDUs of PRF file.
#    prfDimX : int
#        size of PRFx
#    prfDimY : int
#        size of PRFy
#
#    """

