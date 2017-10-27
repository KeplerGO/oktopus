import pytest
import numpy as np
from ..loss import L1Norm
from ..prior import GaussianPrior


@pytest.mark.parametrize("data", ([np.random.normal(size=200)],
                                  [np.random.poisson(size=200)]))
def test_L1Norm_median_estimate(data):
    l1norm = L1Norm(data=data, model=lambda t: t)
    result = l1norm.fit(x0=(np.mean(data)), method='L-BFGS-B')
    assert abs(result.x - np.median(data)) / np.median(data) < 1e-1

@pytest.mark.parametrize("data", ([np.random.normal(size=200)],
                                  [np.random.poisson(size=200)]))
def test_L1Norm_median_estimate_w_regularization(data):
    l1norm = L1Norm(data=data, model=lambda t: t,
                    regularization=GaussianPrior(mean=np.median(data),
                                                 var=np.std(data)**2))
    result = l1norm.fit(x0=(np.mean(data), 0.5), method='L-BFGS-B')
    assert abs(result.x[0] - np.median(data)) / np.median(data) < 1e-1
