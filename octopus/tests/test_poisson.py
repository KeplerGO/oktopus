import pytest
import autograd.numpy as npa
from math import sqrt
import numpy as np
from ..core import PoissonLikelihood

@pytest.mark.parametrize("toy_data",
                         ([np.random.randint(1, 20, size=100)],
                          [np.random.randint(1, 10, size=50)]))
def test_fit(toy_data):
    mean = lambda l: npa.array([l])
    logL = PoissonLikelihood(data=toy_data, mean=mean)
    mean_hat = logL.fit(x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, np.mean(toy_data), decimal=4)
    np.testing.assert_almost_equal(logL.uncertainties(), sqrt(np.mean(toy_data)), decimal=4)
