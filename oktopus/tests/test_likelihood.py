import pytest
import autograd.numpy as npa
from math import sqrt
import numpy as np
from .helpers import ConstantModel, LineModel
from ..likelihood import (MultinomialLikelihood, PoissonLikelihood,
                          GaussianLikelihood, MultivariateGaussianLikelihood,
                          LaplacianLikelihood)
from ..models import WhiteNoiseKernel


@pytest.mark.parametrize("counts, ans, opt_kwargs",
        ([np.array([30, 30]), 0.5, {'optimizer': 'minimize', 'x0': 0.3, 'method': 'Nelder-Mead'}],
         [np.array([90, 10]), 0.9, {'optimizer': 'minimize', 'x0': 0.8, 'method': 'Nelder-Mead'}],
         [np.array([30, 30]), 0.5, {'optimizer': 'differential_evolution', 'bounds': [(0, 1)], 'tol': 1e-8}],
         [np.array([90, 10]), 0.9, {'optimizer': 'differential_evolution', 'bounds': [(0, 1)], 'tol': 1e-8}]))
def test_multinomial_likelihood(counts, ans, opt_kwargs):
    ber_pmf = lambda p: npa.array([p, 1 - p])
    logL = MultinomialLikelihood(data=counts, mean=ber_pmf)
    p_hat = logL.fit(**opt_kwargs)
    np.testing.assert_almost_equal(logL.uncertainties(p_hat.x),
                                   sqrt(p_hat.x[0] * (1 - p_hat.x[0]) / counts.sum()))
    assert abs(p_hat.x - ans) < 0.05
    # analytical jeffrey's prior
    neg_log_jeff_prior = 0.5 * (np.log(p_hat.x) + np.log(1 - p_hat.x) - np.log(counts.sum()))
    np.testing.assert_almost_equal(neg_log_jeff_prior, logL.jeffreys_prior(p_hat.x))
    np.testing.assert_almost_equal(logL.gradient(p_hat.x), 0., decimal=2)


@pytest.mark.parametrize("toy_data, optimizer, model",
                         ([np.random.randint(1, 20, size=100), 'basinhopping', ConstantModel()],
                          [np.random.randint(1, 20, size=100), 'minimize', ConstantModel().evaluate],
                          [np.random.randint(1, 10, size=50), 'basinhopping', ConstantModel().evaluate],
                          [np.random.randint(1, 10, size=50), 'minimize', ConstantModel()]))
def test_poisson_likelihood(toy_data, optimizer, model):
    logL = PoissonLikelihood(data=toy_data, mean=model)
    mean_hat = logL.fit(optimizer=optimizer, x0=np.median(toy_data))
    np.testing.assert_almost_equal(mean_hat.x, np.mean(toy_data), decimal=4)
    np.testing.assert_almost_equal(logL.uncertainties(mean_hat.x),
                                   sqrt(np.mean(toy_data)), decimal=4)
    # test gradients
    l = np.linspace(1, 10, len(toy_data))
    true_grad = np.sum((1 - toy_data / model(l)))
    np.testing.assert_almost_equal(true_grad, logL.gradient([l]))
    np.testing.assert_almost_equal(logL.gradient(mean_hat.x), 0., decimal=3)


@pytest.mark.parametrize("optimizer", ("basinhopping", "minimize"))
def test_gaussian_likelihood(optimizer):
    x = npa.linspace(-5, 5, 20)
    fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    line = LineModel(x)
    logL = GaussianLikelihood(fake_data, line, 4)
    p0 = (1, 1) # dumb initial_guess for slope and intercept
    p_hat = logL.fit(optimizer=optimizer, x0=p0)
    # lsq solution from linear algebra
    M = np.array([[np.sum(x * x), np.sum(x)], [np.sum(x), len(x)]])
    p_hat_linalg = np.dot(np.linalg.inv(M),
                          np.array([np.sum(fake_data * x), np.sum(fake_data)]))
    np.testing.assert_almost_equal(p_hat.x, p_hat_linalg, decimal=4)
    np.testing.assert_almost_equal(logL.gradient(p_hat.x), 0., decimal=3)

    # tests gradients
    a = np.random.rand()
    b = np.random.rand()
    true_grad = - np.sum((fake_data - line(a, b)) * np.array([x, np.ones(len(x))]) / 4, axis=1)
    np.testing.assert_almost_equal(true_grad, logL.gradient([a, b]))

    # test that MultivariateGaussianLikelihood returns the previous result for
    # a WhiteNoiseKernel
    logL = MultivariateGaussianLikelihood(data=fake_data, mean=line,
                                          cov=WhiteNoiseKernel(n=len(fake_data)),
                                          dim=2)
    p0 = (1, 1, 2)
    p_hat = logL.fit(x0=p0)
    np.testing.assert_almost_equal(p_hat.x[:2], p_hat_linalg, decimal=4)


@pytest.mark.parametrize("data", ([np.random.normal(size=200)],
                                  [np.random.poisson(size=200)]))
def test_laplacian_likelihood(data):
    l1norm = LaplacianLikelihood(data=data, mean=lambda t: t, var=1)
    result = l1norm.fit(x0=(np.mean(data)), method='Nelder-Mead')
    assert abs(result.x - np.median(data)) / np.median(data) < 1e-1
