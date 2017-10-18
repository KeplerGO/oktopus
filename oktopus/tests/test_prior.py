import pytest
import numpy as np
from numpy.testing import assert_array_equal
from ..prior import UniformPrior, GaussianPrior, JointPrior


def test_uniform_prior():
    unif = UniformPrior(-.5, .5)
    x = np.array([-.25, .25, 0])

    assert np.isfinite(unif(x))
    assert not np.isfinite(unif(x + 1))

def test_joint_uniform_priors():
    unif = JointPrior(UniformPrior(-.5, .5), UniformPrior(.5, 1.))
    assert_array_equal(unif.mean, [0, .75])
    assert np.isfinite(unif((.4999, .5)))
    assert ~np.isfinite(unif((.5, .5)))

def test_gaussian_prior():
    gauss = GaussianPrior(0, 1)
    assert gauss(0) == 0.0
    assert gauss(4) == 8.0

def test_joint_gaussian_priors():
    gauss = JointPrior(GaussianPrior(0, 1), GaussianPrior(1, 1))
    assert gauss((0, 1)) == 0.0
    assert gauss((2, 3)) == 4.0
    assert_array_equal(gauss.mean, [0, 1])

def test_joint_mixed_prior():
    unif = UniformPrior(-1, 1)
    gauss = GaussianPrior(0, 1)
    jp = JointPrior(unif, gauss)

    assert_array_equal(jp.mean, [0, 0])

    assert jp.evaluate((.5, .5)) == unif.evaluate(.5) + gauss.evaluate(.5)
    assert not np.isfinite(jp((1.5, .5)))

    jp = JointPrior(unif, unif, gauss, gauss)
    assert jp.evaluate((.5, .5, .5, .5)) == 2 * (unif.evaluate(.5) + gauss.evaluate(.5))
