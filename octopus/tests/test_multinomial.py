import pytest
import numpy as np
from ..core import MultinomialLikelihood

@pytest.mark.parametrize("counts, p0, ans",
                         ([np.array([20, 30]), 0.5, 0.4],
                          [np.array([30, 30]), 0.5, 0.5],
                          [np.array([30, 20]), 0.9, 0.6],
                          [np.array([99, 1]),  0.7, 0.99]))
def test_fit(counts, p0, ans):
    ber_pmf = lambda p: np.array([p, 1 - p])
    logL = MultinomialLikelihood(data=counts, pmf=ber_pmf)
    p_hat = logL.fit(x0=p0)
    #p_hat_unc = logL.uncertainties()
    np.testing.assert_almost_equal(p_hat.x[0], ans, decimal=4)
