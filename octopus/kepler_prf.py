from .models import get_initial_guesses
from .core import PoissonLikelihood
from abc import ABC, abstractmethod
import scipy
import pandas as pd
import math
from pyke import TargetPixelFile


__all__ = ['KeplerPRFPhotometry', 'KeplerPRF']


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
        self.opt_params = []
        self.residuals = []
        self.uncertainties = []

    def do_photometry(self, tpf, initial_guesses=None):
        if initial_guesses is None:
            # this must be clever enough to find the number of stars
            # great way to go is to use photutils.detection.DAOStarFinder
            initial_guesses, _ = get_inital_guesses(tpf.flux)

        for t in range(len(tpf.time)):
            logL = self.loss_function(tpf.flux, self.prf_model)
            opt_result = logL.fit(initial_guesses).x
            residuals_opt_result = tpf.flux - self.prf_model(*opt_result.x)
            self.opt_params.append(opt_result.x)
            self.residuals.append(residuals_opt_result)
            self.uncertainties.append(logL.uncertainties())

        self.opt_params = self.opt_params.reshape((tpf.shape[0], len(initial_guesses)))
        self.uncertainties = self.uncertainties.reshape((tpf.shape[0], len(initial_guesses)))

    def generate_residuals_movie(self):
        pass


class KeplerPRF(KeplerTargetPixelFile):
    """
    Kepler's Pixel Response Function

    This class provides the necessary interface to load Kepler PSF
    calibration files and to create a model that can be fit as a function
    of flux and centroid position.
    """

    def __init__(self, prf_files_dir):
        self.prf_files_dir = prf_files_dir

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
        return self.prf_to_detector(F, xo, yo)

    def __call__(self, F, xo, yo):
        return self.evaluate(F, xo, yo)

    def read_prf_calibration_file(self, path, ext):
        prf_cal_file = pyfits.open(path)
        data = prf_cal_file[ext].data
        #crpix1p = prf_cal_file[ext].header['CRPIX1P']
        #crpix2p = prf_cal_file[ext].header['CRPIX2P']
        # looks like these data below are the same for all prf calibration files
        crval1p = prf_cal_file[ext].header['CRVAL1P']
        crval2p = prf_cal_file[ext].header['CRVAL2P']
        cdelt1p = prf_cal_file[ext].header['CDELT1P']
        cdelt2p = prf_cal_file[ext].header['CDELT2P']
        prf_cal_file.close()
        #return data, crpix1p, crpix2p, crval1p, crval2p, cdelt1p, cdelt2p
        return data, crval1p, crval2p, cdelt1p, cdelt2p

    def prepare_prf(self):
        n_hdu = 5
        min_prf_weight = 1e-6
        # determine suitable PRF calibration file
        if int(module) < 10:
            prefix = 'kplr0'
        else:
            prefix = 'kplr'
        prf_file_path = os.path.join(self.prf_files_dir,
                                     prefix + module + '.' + output + '*_prf.fits')
        prffile = glob.glob(prf_file_path)[0]

        # read PRF images
        prfn = [0] * n_hdu
        crval1p = np.zeros(n_hdu, dtype='float32')
        crval2p = np.zeros(n_hdu, dtype='float32')
        cdelt1p = np.zeros(n_hdu, dtype='float32')
        cdelt2p = np.zeros(n_hdu, dtype='float32')
        for i in range(1, n_hdu+1)::/
            prfn[i], crval1p[i], crval2p[i], cdelt1p[i], cdelt2p[i] = self.read_prf_calibration_file(prffile, i)
        prfn = np.array(prfn)
        PRFx = np.arange(0.5, np.shape(prfn[0])[1] + 0.5)
        PRFy = np.arange(0.5, np.shape(prfn[0])[0] + 0.5)
        PRFx = (PRFx - np.size(PRFx) / 2) * cdelt1p[0]
        PRFy = (PRFy - np.size(PRFy) / 2) * cdelt2p[0]

        # interpolate the calibrated PRF shape to the target position
        prf = np.zeros(np.shape(prfn[0]), dtype='float32')
        prfWeight = np.zeros(n_hdu, dtype='float32')
        ref_column = column + (xdim - 1.) / 2.
        ref_row = row + (ydim - 1.) / 2.
        for i in range(n_hdu):
            prfWeight[i] = math.sqrt((ref_column - crval1p[i]) ** 2
                                     + (ref_row - crval2p[i]) ** 2))
            if prfWeight[i] < minimum_prf_weight:
                prfWeight[i] = minimum_prf_weight
            prf += prfn[i] / prfWeight[i]
        prf /= (np.nansum(prf) * cdelt1p[0] * cdelt2p[0])

        # location of the data image centered on the PRF image (in PRF pixel units)
        prfDimY = int(ydim / cdelt1p[0])
        prfDimX = int(xdim / cdelt2p[0])
        PRFy0 = int(np.round((np.shape(prf)[0] - prfDimY) / 2))
        PRFx0 = int(np.round((np.shape(prf)[1] - prfDimX) / 2))
        DATx = np.arange(column, column + xdim)
        DATy = np.arange(row, row + ydim)
        splineInterpolation = scipy.interpolate.RectBivariateSpline(PRFx, PRFy, prf)

        return (splineInterpolation, DATx, DATy, prf, PRFx, PRFy, PRFx0, PRFy0,
                cdelt1p, cdelt2p, prfDimX, prfDimY)
