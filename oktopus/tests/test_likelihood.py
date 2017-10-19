import pytest
import autograd.numpy as npa
from math import sqrt
import numpy as np
from ..likelihood import (MultinomialLikelihood, PoissonLikelihood,
                          GaussianLikelihood, MultivariateGaussianLikelihood)
from ..models import WhiteNoiseKernel


@pytest.mark.parametrize("counts, ans, opt_kwargs",
        ([np.array([30, 30]), 0.5, {'optimizer': 'minimize', 'x0': 0.3, 'method': 'Nelder-Mead'}],
         [np.array([90, 10]), 0.9, {'optimizer': 'minimize', 'x0': 0.8, 'method': 'Nelder-Mead'}],
         [np.array([30, 30]), 0.5, {'optimizer': 'differential_evolution', 'bounds': [(0, 1)]}],
         [np.array([90, 10]), 0.9, {'optimizer': 'differential_evolution', 'bounds': [(0, 1)]}]))
def test_multinomial_likelihood(counts, ans, opt_kwargs):
    ber_pmf = lambda p: npa.array([p, 1 - p])
    logL = MultinomialLikelihood(data=counts, mean=ber_pmf)
    p_hat = logL.fit(**opt_kwargs)
    np.testing.assert_almost_equal(logL.uncertainties(p_hat.x),
                                   sqrt(p_hat.x[0] * (1 - p_hat.x[0]) / counts.sum()))
    assert abs(p_hat.x - ans) < 0.05
    neg_log_jeff_prior = 0.5 * (np.log(p_hat.x) + np.log(1 - p_hat.x) - np.log(counts.sum()))
    np.testing.assert_almost_equal(neg_log_jeff_prior, logL.jeffreys_prior(p_hat.x))


@pytest.mark.parametrize("toy_data, optimizer",
                         ([np.random.randint(1, 20, size=100), 'basinhopping'],
                          [np.random.randint(1, 20, size=100), 'minimize'],
                          [np.random.randint(1, 10, size=50), 'basinhopping'],
                          [np.random.randint(1, 10, size=50), 'minimize']))
def test_poisson_likelihood(toy_data, optimizer):
    mean = lambda l: npa.array([l])
    logL = PoissonLikelihood(data=toy_data, mean=mean)
    mean_hat = logL.fit(optimizer=optimizer, x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, np.mean(toy_data), decimal=4)
    np.testing.assert_almost_equal(logL.uncertainties(mean_hat.x),
                                   sqrt(np.mean(toy_data)), decimal=4)


@pytest.mark.parametrize("optimizer", ("basinhopping", "minimize"))
def test_gaussian_likelihood(optimizer):
    x = npa.linspace(-5, 5, 20)
    np.random.seed(0)
    fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    def line(alpha, beta):
        return alpha * x + beta
    logL = GaussianLikelihood(fake_data, line, 4)
    p0 = (1, 1) # dumb initial_guess for alpha and beta
    p_hat = logL.fit(optimizer=optimizer, x0=p0)
    # lsq solution from linear algebra
    M = np.array([[np.sum(x * x), np.sum(x)], [np.sum(x), len(x)]])
    p_hat_linalg = np.dot(np.linalg.inv(M),
                          np.array([np.sum(fake_data * x), np.sum(fake_data)]))
    np.testing.assert_almost_equal(p_hat.x, p_hat_linalg, decimal=4)

    # test that MultivariateGaussianLikelihood returns the previous result for
    # a WhiteNoiseKernel
    logL = MultivariateGaussianLikelihood(data=fake_data, mean=line,
                                          cov=WhiteNoiseKernel(n=len(fake_data)),
                                          dim=2)
    p0 = (1, 1, 2)
    p_hat = logL.fit(x0=p0)
    np.testing.assert_almost_equal(p_hat.x[:2], p_hat_linalg, decimal=4)
