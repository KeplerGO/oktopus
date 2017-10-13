import pytest
import autograd.numpy as npa
from math import sqrt
import numpy as np
from ..likelihood import (MultinomialLikelihood, PoissonLikelihood,
                          GaussianLikelihood, MultivariateGaussianLikelihood)
from ..models import WhiteNoiseKernel


@pytest.mark.parametrize("counts, p0, ans",
                         ([np.array([20, 30]), 0.5, 0.4],
                          [np.array([30, 30]), 0.5, 0.5],
                          [np.array([30, 20]), 0.9, 0.6],
                          [np.array([80, 20]), 0.7, 0.8]))
def test_multinomial_likelihood(counts, p0, ans):
    ber_pmf = lambda p: npa.array([p, 1 - p])
    logL = MultinomialLikelihood(data=counts, mean=ber_pmf)
    p_hat = logL.fit(x0=p0)
    np.testing.assert_almost_equal(logL.uncertainties(p_hat.x),
                                   sqrt(p_hat.x[0] * (1 - p_hat.x[0]) / counts.sum()))
    np.testing.assert_almost_equal(p_hat.x, ans, decimal=4)
    neg_log_jeff_prior = 0.5 * (np.log(p_hat.x) + np.log(1 - p_hat.x) - np.log(counts.sum()))
    np.testing.assert_almost_equal(neg_log_jeff_prior, logL.jeffreys_prior(p_hat.x))
    np.testing.assert_almost_equal(logL.gradient(p_hat.x), 0., decimal=2)


@pytest.mark.parametrize("toy_data",
                         ([np.random.randint(1, 20, size=100)],
                          [np.random.randint(1, 10, size=50)]))
def test_poisson_likelihood(toy_data):
    mean = lambda l: npa.array([l])
    logL = PoissonLikelihood(data=toy_data, mean=mean)
    mean_hat = logL.fit(x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, np.mean(toy_data), decimal=4)
    np.testing.assert_almost_equal(logL.uncertainties(mean_hat.x),
                                   sqrt(np.mean(toy_data)), decimal=4)
    np.testing.assert_almost_equal(logL.gradient(mean_hat.x), 0., decimal=3)

def test_gaussian_likelihood():
    x = npa.linspace(-5, 5, 20)
    np.random.seed(0)
    fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    def line(alpha, beta):
        return alpha * x + beta
    logL = GaussianLikelihood(fake_data, line, 4)
    p0 = (1, 1) # dumb initial_guess for alpha and beta
    p_hat = logL.fit(x0=p0)
    # lsq solution from linear algebra
    M = np.array([[np.sum(x * x), np.sum(x)], [np.sum(x), len(x)]])
    p_hat_linalg = np.dot(np.linalg.inv(M),
                          np.array([np.sum(fake_data * x), np.sum(fake_data)]))
    np.testing.assert_almost_equal(p_hat.x, p_hat_linalg, decimal=4)
    np.testing.assert_almost_equal(logL.gradient(p_hat.x), 0., decimal=3)

    # test that MultivariateGaussianLikelihood returns the previous result for
    # a WhiteNoiseKernel
    logL = MultivariateGaussianLikelihood(data=fake_data, mean=line,
                                          cov=WhiteNoiseKernel(n=len(fake_data)),
                                          dim=2)
    p0 = (1, 1, 2)
    p_hat = logL.fit(x0=p0)
    np.testing.assert_almost_equal(p_hat.x[:2], p_hat_linalg, decimal=4)
