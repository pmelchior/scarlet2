# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103
# ruff: noqa: D106

import numpyro.distributions as dist
import pytest
from numpyro.infer.initialization import init_to_sample

from scarlet2 import Parameter, Parameters
from scarlet2.infer import fit
from scarlet2.validation_utils import (
    set_validation,
)


@pytest.fixture(autouse=True)
def setup_validation():
    """Automatically disable validation for all tests. This permits the creation
    of intentionally invalid Observation objects."""
    set_validation(False)


def test_fit(scene, good_obs):
    max_iter = 10
    fit(scene, good_obs, max_iter=max_iter, progress_bar=False)


def test_sample(scene, good_obs):
    with Parameters(scene):  # new parameters: only the one spectrum gets sampled
        p = scene.sources[0].spectrum
        prior = dist.Normal(p, scale=1).to_event(1)
        Parameter(p, name="spectrum:0", prior=prior)
    init_strategy = init_to_sample
    scene.sample(
        good_obs,
        num_samples=10,
        num_warmup=10,
        dense_mass=True,
        init_strategy=init_strategy,
        progress_bar=False,
    )
