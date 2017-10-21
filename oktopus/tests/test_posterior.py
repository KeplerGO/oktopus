import pytest
import autograd.numpy as npa
from math import sqrt
import numpy as np
from ..likelihood import PoissonLikelihood
from ..posterior import Posterior, GaussianPosterior, PoissonPosterior
from ..prior import UniformPrior, GaussianPrior


@pytest.mark.parametrize("toy_data",
                         ([np.random.randint(1, 20, size=100)],
                          [np.random.randint(1, 10, size=50)]))
def test_posterior(toy_data):
    mean = lambda l: npa.array([l])
    unif_prior = UniformPrior(lb=np.min(toy_data), ub=np.max(toy_data))
    logP = PoissonPosterior(data=toy_data, mean=mean, prior=unif_prior)
    mean_hat = logP.fit(x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, np.mean(toy_data), decimal=4)
    np.testing.assert_almost_equal(logP.gradient(mean_hat.x), 0, decimal=3)

    # test that creating a Posterior object results in the same estimations
    logP_ = Posterior(likelihood=PoissonLikelihood(data=toy_data, mean=mean),
                      prior=unif_prior)
    mean_hat_ = logP_.fit(x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, mean_hat_.x)
    np.testing.assert_almost_equal(logP_.gradient(mean_hat_.x),
                                   logP.gradient(mean_hat.x))

    # Using a Gaussian prior
    gauss_prior = GaussianPrior(mean=np.mean(toy_data), var=np.mean(toy_data))
    logP.prior = gauss_prior
    mean_hat = logP.fit(x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, np.mean(toy_data), decimal=4)
    np.testing.assert_almost_equal(logP.gradient(mean_hat.x), 0, decimal=3)

    # Maybe we can approximate the count data as a Gaussian distribution
    logP = GaussianPosterior(data=toy_data, mean=mean, var=np.std(toy_data) ** 2, prior=unif_prior)
    mean_hat = logP.fit(x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, np.mean(toy_data), decimal=4)
    np.testing.assert_almost_equal(logP.gradient(mean_hat.x), 0, decimal=3)
    logP.prior = gauss_prior
    mean_hat = logP.fit(x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, np.mean(toy_data), decimal=4)
    np.testing.assert_almost_equal(logP.gradient(mean_hat.x), 0, decimal=3)
