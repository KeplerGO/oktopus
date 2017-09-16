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

class KeplerPRF(object):

    def __init__(self):
        pass

    def prf_to_detector(self):
        pass

    def prf(self):
        pass

    def interpolate(self):
        pass

#def PRF2DET(flux, OBJx, OBJy, DATx, DATy, wx, wy, a, splineInterpolation):
#    """
#    PRF interpolation function
#    """
#
#    # trigonometry
#    cosa = np.cos(radians(a))
#    sina = np.sin(radians(a))
#
#    # where in the pixel is the source position?
#    PRFfit = np.zeros((np.size(DATy), np.size(DATx)))
#    for i in range(len(flux)):
#        FRCx, INTx = modf(OBJx[i])
#        FRCy, INTy = modf(OBJy[i])
#        if FRCx > 0.5:
#            FRCx -= 1.0
#            INTx += 1.0
#        if FRCy > 0.5:
#            FRCy -= 1.0
#            INTy += 1.0
#        FRCx = -FRCx
#        FRCy = -FRCy
#
#        # constuct model PRF in detector coordinates
#        for (j, y) in enumerate(DATy):
#            for (k, x) in enumerate(DATx):
#                xx = x - INTx + FRCx
#                yy = y - INTy + FRCy
#                dx = xx * cosa - yy * sina
#                dy = xx * sina + yy * cosa
#                PRFfit[j, k] = PRFfit[j, k] + splineInterpolation(dy * wy, dx * wx) * flux[i]
#
#    return PRFfit
#
#def PRF(params, *args):
#    """
#    PRF model
#    """
#    # arguments
#    DATx = args[0]
#    DATy = args[1]
#    DATimg = args[2]
#    DATerr = args[3]
#    nsrc = args[4]
#    splineInterpolation = args[5]
#    col = args[6]
#    row = args[7]
#
#    # parameters
#    f = np.empty((nsrc))
#    x = np.empty((nsrc))
#    y = np.empty((nsrc))
#    for i in range(nsrc):
#        f[i] = params[i]
#        x[i] = params[nsrc + i]
#        y[i] = params[nsrc * 2 + i]
#
#    # calculate PRF model binned to the detector pixel size
#    PRFfit = PRF2DET(f,x,y,DATx,DATy,1.0,1.0,0.0,splineInterpolation)
#    # calculate the sum squared difference between data and model
#    PRFres = np.nansum(np.square(DATimg - PRFfit))
#    # keep the fit centered
#    if max(abs(col - x[0]), abs(row - y[0])) > 10.0:
#        PRFres = 1.0e300
#    return PRFres
#
#
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
#    n_hdu = 5
#    minimum_prf_weight = 1.e-6
#
#    # determine suitable PRF calibration file
#    if int(module) < 10:
#        prefix = 'kplr0'
#    else:
#        prefix = 'kplr'
#    prfglob = os.path.join(prfdir, prefix + module + '.' + output + '*_prf.fits')
#    try:
#        prffile = glob.glob(prfglob)[0]
#    except:
#        errmsg = "ERROR -- KEPPRF: No PRF file found in {0}".format(prfdir)
#        kepmsg.err(logfile, errmsg, verbose)
#
#    # read PRF images
#    prfn = [0] * n_hdu
#    crval1p = np.zeros(n_hdu, dtype='float32')
#    crval2p = np.zeros(n_hdu, dtype='float32')
#    cdelt1p = np.zeros(n_hdu, dtype='float32')
#    cdelt2p = np.zeros(n_hdu, dtype='float32')
#    for i in range(n_hdu):
#        (prfn[i], _, _, crval1p[i], crval2p[i], cdelt1p[i], cdelt2p[i]) = \
#            kepio.readPRFimage(prffile, i+1, logfile, verbose)
#    prfn = np.array(prfn)
#    PRFx = np.arange(0.5, np.shape(prfn[0])[1] + 0.5)
#    PRFy = np.arange(0.5, np.shape(prfn[0])[0] + 0.5)
#    PRFx = (PRFx - np.size(PRFx) / 2) * cdelt1p[0]
#    PRFy = (PRFy - np.size(PRFy) / 2) * cdelt2p[0]
#
#    # interpolate the calibrated PRF shape to the target position
#    prf = np.zeros(np.shape(prfn[0]), dtype='float32')
#    prfWeight = np.zeros(n_hdu, dtype='float32')
#    ref_column = column + (xdim - 1.) / 2.
#    ref_row = row + (ydim - 1.) / 2.
#    for i in range(n_hdu):
#        prfWeight[i] = math.sqrt(
#            (ref_column - crval1p[i])**2 + (ref_row - crval2p[i])**2)
#        if prfWeight[i] < minimum_prf_weight:
#            prfWeight[i] = minimum_prf_weight
#        prf += prfn[i] / prfWeight[i]
#    prf /= (np.nansum(prf) * cdelt1p[0] * cdelt2p[0])
#
#    # location of the data image centered on the PRF image (in PRF pixel units)
#    prfDimY = int(ydim / cdelt1p[0])
#    prfDimX = int(xdim / cdelt2p[0])
#    PRFy0 = int(np.round((np.shape(prf)[0] - prfDimY) / 2))
#    PRFx0 = int(np.round((np.shape(prf)[1] - prfDimX) / 2))
#    DATx = np.arange(column, column + xdim)
#    DATy = np.arange(row, row + ydim)
#
#    # interpolation function over the PRF
#    splineInterpolation = RectBivariateSpline(PRFx, PRFy, prf)
#
#    return (splineInterpolation, DATx, DATy, prf, PRFx, PRFy, PRFx0, PRFy0,
#            cdelt1p, cdelt2p, prfDimX, prfDimY)
