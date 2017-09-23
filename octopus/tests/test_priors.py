import pytest
import numpu
from ..core import UniformPrior, GaussianPrior, JointPrior

def test_joint_prior():
    unif = UniformPrior(-1, 1)
    gauss = GaussianPrior(0, 1)
    jp = JointPrior(unif, gauss)
    assert jp.evaluate((.5, .5)) == unif.evaluate(.5) + gauss.evaluate(.5)
