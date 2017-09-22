import os
import numpy as np
from astropy.io import fits

from .. import models
from .. import core
from ..kepler_prf import KeplerPRF

PRFDIR = os.path.expanduser('~/.pyke/kepler-prf-calibration-data/')


def test_prf_integration():
    """Does the PRF model integrate to the requested flux?"""
    channel, col, row = 84, 200, 800
    shape = (18, 14)
    flux = 100
    prf = KeplerPRF(prf_files_dir=PRFDIR, channel=channel, column=col, row=row, shape=shape)
    prf_sum = prf(flux, col + shape[0]/2, row + shape[1]/2, 0).sum()
    assert np.isclose(prf_sum, flux, rtol=0.1)


def test_prf_vs_aperture_photometry():
    """Is the PRF photometry result consistent with simple aperture photometry?"""
    tpf = fits.open("data/ktwo201907706-c01-first-cadence.fits.gz")
    col, row = 173, 526
    prf = KeplerPRF(prf_files_dir=PRFDIR, channel=tpf[0].header['CHANNEL'],
                    column=col, row=row,
                    shape=tpf[1].data.shape)
    fluxo, xo, yo, bkgo = models.get_initial_guesses(data=tpf[1].data,
                                                     X=prf.x,
                                                     Y=prf.y)
    aperture_flux = tpf[1].data.sum()
    logL = core.PoissonLikelihood(tpf[1].data, prf.evaluate)
    fitresult = logL.fit((fluxo, xo, yo, bkgo))
    prf_flux = fitresult.x[0]
    prf_bkg = fitresult.x[3]
    assert np.isclose(prf_bkg, np.percentile(tpf[1].data, 10), rtol=0.1)
    assert np.isclose(aperture_flux, prf_flux, rtol=0.1)
