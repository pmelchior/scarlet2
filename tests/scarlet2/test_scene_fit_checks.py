from functools import partial
from unittest.mock import patch

import jax.numpy as jnp
import pytest
from huggingface_hub import hf_hub_download
from numpyro.distributions import constraints
from scarlet2 import init
from scarlet2.frame import Frame
from scarlet2.module import Parameter, relative_step
from scarlet2.observation import Observation
from scarlet2.psf import ArrayPSF
from scarlet2.scene import FitValidator, Scene
from scarlet2.source import PointSource, Source
from scarlet2.validation_utils import (
    ValidationError,
    ValidationInfo,
    ValidationWarning,
    set_validation,
)


@pytest.fixture(autouse=True)
def setup_validation():
    """Automatically disable validation for all tests. This permits the creation
    of intentionally invalid Observation objects."""
    set_validation(False)


@pytest.fixture()
def data_file():
    """Download and load a realistic test file. This is the same data used in the
    quickstart notebook. The data will be manipulated to create invalid inputs for
    the `bad_obs` fixture."""
    filename = hf_hub_download(
        repo_id="astro-data-lab/scarlet-test-data", filename="hsc_cosmos_35.npz", repo_type="dataset"
    )
    return jnp.load(filename)


@pytest.fixture()
def good_obs(data_file):
    """Create an observation that should pass all validation checks."""
    data = jnp.asarray(data_file["images"])
    channels = [str(f) for f in data_file["filters"]]
    weights = jnp.asarray(1 / data_file["variance"])
    psf = jnp.asarray(data_file["psfs"])

    return Observation(
        data=data,
        weights=weights,
        channels=channels,
        psf=ArrayPSF(psf),
    )


@pytest.fixture()
def scene(good_obs, data_file):
    """Assemble a scene from the good observation and the data file."""
    model_frame = Frame.from_observations(good_obs)
    centers = jnp.array([(src["y"], src["x"]) for src in data_file["catalog"]])  # Note: y/x convention!

    with Scene(model_frame) as scene:
        for i, center in enumerate(centers):
            if i == 0:  # we know source 0 is a star
                spectrum = init.pixel_spectrum(good_obs, center, correct_psf=True)
                PointSource(center, spectrum)
            else:
                try:
                    spectrum, morph = init.from_gaussian_moments(good_obs, center, min_corr=0.99)
                except ValueError:
                    spectrum = init.pixel_spectrum(good_obs, center)
                    morph = init.compact_morphology()

                Source(center, spectrum, morph)

    return scene


@pytest.fixture()
def parameters(scene, good_obs):
    """Create parameters for the scene."""
    spec_step = partial(relative_step, factor=0.05)
    morph_step = partial(relative_step, factor=1e-3)

    parameters = scene.make_parameters()
    for i in range(len(scene.sources)):
        parameters += Parameter(
            scene.sources[i].spectrum,
            name=f"spectrum:{i}",
            constraint=constraints.positive,
            stepsize=spec_step,
        )
        if i == 0:
            parameters += Parameter(scene.sources[i].center, name=f"center:{i}", stepsize=0.1)
        else:
            parameters += Parameter(
                scene.sources[i].morphology,
                name=f"morph:{i}",
                constraint=constraints.unit_interval,
                stepsize=morph_step,
            )

    scene.set_spectra_to_match(good_obs, parameters)
    return parameters


@pytest.mark.parametrize(
    "mocked_chi_value,expected",
    [
        (0.1, ValidationInfo),
        (2.0, ValidationWarning),
        (10.0, ValidationError),
    ],
)
def test_check_goodness_of_fit(scene, parameters, good_obs, mocked_chi_value, expected):
    """Mocked goodness_of_fit return to produces the expected ValidationResult type"""
    scene_ = scene.fit(good_obs, parameters, max_iter=1, e_rel=1e-4, progress_bar=True)

    checker = FitValidator(scene_, good_obs)

    with patch.object(type(good_obs), "goodness_of_fit", return_value=mocked_chi_value) as _:
        results = checker.check_goodness_of_fit()

    assert isinstance(results, expected)


@pytest.mark.parametrize(
    "mocked_chi_value,expected",
    [
        (0.1, ValidationInfo),
        (2.0, ValidationWarning),
        (10.0, ValidationError),
    ],
)
def test_check_chi_squared_in_box_and_border(scene, parameters, good_obs, mocked_chi_value, expected):
    """Mocked chi-squared evaluation in box and border."""

    scene_ = scene.fit(good_obs, parameters, max_iter=1, e_rel=1e-4, progress_bar=True)
    checker = FitValidator(scene_, good_obs)
    mock_return = {0: {"in": mocked_chi_value, "out": mocked_chi_value}}

    with patch.object(type(good_obs), "eval_chi_square_in_box_and_border", return_value=mock_return) as _:
        results = checker.check_chi_square_in_box_and_border()

    assert all(isinstance(res, expected) for res in results)
