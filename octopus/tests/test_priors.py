import pytest
import numpy as np
from ..core import UniformPrior, GaussianPrior, JointPrior


def test_uniform_prior():
    unif = UniformPrior(-.5, .5)
    x = np.array([-.25, .25, 0])

    assert np.isfinite(unif(x))
    assert not np.isfinite(unif(x + 1))

def test_add_uniform_priors():
    unif = UniformPrior(-.5, .5) + UniformPrior(.5, 1.)
    assert isinstance(unif, UniformPrior)
    assert np.isfinite(unif((.4999, .5)))
    assert ~np.isfinite(unif((.5, .5)))

    unif = unif + UniformPrior(2., 3.)
    assert isinstance(unif, UniformPrior)
    assert np.isfinite(unif((.4999, .5, 2.5)))
    assert ~np.isfinite(unif((.5, .5, .0)))


def test_joint_prior():
    unif = UniformPrior(-1, 1)
    gauss = GaussianPrior(0, 1)
    jp = JointPrior(unif, gauss)

    assert jp.evaluate((.5, .5)) == unif.evaluate(.5) + gauss.evaluate(.5)
    assert not np.isfinite(jp((1.5, .5)))
