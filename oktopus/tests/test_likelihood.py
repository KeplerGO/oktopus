import pytest
import autograd.numpy as npa
from math import sqrt
import numpy as np
from ..likelihood import MultinomialLikelihood, PoissonLikelihood


@pytest.mark.parametrize("counts, p0, ans",
                         ([np.array([20, 30]), 0.5, 0.4],
                          [np.array([30, 30]), 0.5, 0.5],
                          [np.array([30, 20]), 0.9, 0.6],
                          [np.array([80, 20]), 0.7, 0.8]))
def test_multinomial_likelihood(counts, p0, ans):
    ber_pmf = lambda p: npa.array([p, 1 - p])
    logL = MultinomialLikelihood(data=counts, mean=ber_pmf)
    p_hat = logL.fit(x0=p0)
    np.testing.assert_almost_equal(logL.uncertainties(),
                                   sqrt(p_hat.x[0] * (1 - p_hat.x[0]) / counts.sum()))
    np.testing.assert_almost_equal(p_hat.x[0], ans, decimal=4)

@pytest.mark.parametrize("toy_data",
                         ([np.random.randint(1, 20, size=100)],
                          [np.random.randint(1, 10, size=50)]))
def test_poisson_likelihood(toy_data):
    mean = lambda l: npa.array([l])
    logL = PoissonLikelihood(data=toy_data, mean=mean)
    mean_hat = logL.fit(x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, np.mean(toy_data), decimal=4)
    np.testing.assert_almost_equal(logL.uncertainties(), sqrt(np.mean(toy_data)), decimal=4)
