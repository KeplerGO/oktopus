import pytest
import autograd.numpy as npa
import numpy as np
from ..core import MultinomialLikelihood

@pytest.mark.parametrize("counts, p0, ans, unc",
                         ([np.array([20, 30]), 0.5, 0.4, 0.06928203],
                          [np.array([30, 30]), 0.5, 0.5, 0.06454972],
                          [np.array([30, 20]), 0.9, 0.6, 0.06928119],
                          [np.array([99, 1]),  0.7, 0.99, 0.00995949]))
def test_fit(counts, p0, ans, unc):
    ber_pmf = lambda p: npa.array([p, 1 - p])
    logL = MultinomialLikelihood(data=counts, mean=ber_pmf)
    p_hat = logL.fit(x0=p0)
    print(logL.uncertainties())
    np.testing.assert_almost_equal(logL.uncertainties(), unc, decimal=4)
    np.testing.assert_almost_equal(p_hat.x[0], ans, decimal=4)
