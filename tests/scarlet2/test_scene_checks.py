from functools import partial

import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from numpyro.distributions import constraints
from scarlet2 import init
from scarlet2.frame import Frame
from scarlet2.module import Parameter, relative_step
from scarlet2.observation import Observation
from scarlet2.psf import ArrayPSF
from scarlet2.scene import Scene, SceneValidator
from scarlet2.source import PointSource, Source
from scarlet2.validation_utils import (
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
    return np.load(filename)


@pytest.fixture()
def bad_obs(data_file):
    """Create an observation that should fail multiple validation checks."""

    data = np.asarray(data_file["images"])
    channels = [str(f) for f in data_file["filters"]]
    weights = np.asarray(1 / data_file["variance"])
    psf = np.asarray(data_file["psfs"])

    weights = weights[:-1]  # Remove the last weight to create a mismatch in dimensions
    weights[0][0] = np.inf  # Set one weight to infinity
    weights[1][0] = -1.0  # Set one weight to a negative value
    psf = psf[:-1]  # Remove the last PSF to create a mismatch in dimensions
    psf = psf[0] + 0.001

    return Observation(
        data=data,
        weights=weights,
        channels=channels,
        psf=ArrayPSF(psf),
    )


@pytest.fixture()
def good_obs(data_file):
    """Create an observation that should pass all validation checks."""
    data = np.asarray(data_file["images"])
    channels = [str(f) for f in data_file["filters"]]
    weights = np.asarray(1 / data_file["variance"])
    psf = np.asarray(data_file["psfs"])

    return Observation(
        data=data,
        weights=weights,
        channels=channels,
        psf=ArrayPSF(psf),
    )


@pytest.fixture()
def full_overlap_scene(good_obs, data_file):
    """Assemble a scene from the good observation and the data file."""
    model_frame = Frame.from_observations(good_obs)
    centers = np.array([(src["y"], src["x"]) for src in data_file["catalog"]])  # Note: y/x convention!

    with Scene(model_frame) as scene:
        spectrum = init.pixel_spectrum(good_obs, centers[0], correct_psf=True)
        PointSource(centers[0], spectrum)

    return scene


@pytest.fixture()
def partial_overlap_scene(good_obs):
    """Assemble a scene from the good observation and the data file."""
    model_frame = Frame.from_observations(good_obs)

    with Scene(model_frame) as scene:
        spectrum = init.pixel_spectrum(good_obs, np.array([1, 1]), correct_psf=True)
        PointSource(np.array([1, 1]), spectrum)

    return scene


@pytest.fixture()
def no_overlap_scene(good_obs):
    """Assemble a scene from the good observation and the data file."""
    model_frame = Frame.from_observations(good_obs)

    with Scene(model_frame) as scene:
        spectrum = init.pixel_spectrum(good_obs, np.array([0, 0]))
        morph = init.compact_morphology()
        Source(np.array([-10, -10]), spectrum, morph)

    return scene


@pytest.fixture()
def parameters(scene):
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


def test_check_source_boxes_comparable_to_observation_source_inside(full_overlap_scene, good_obs):
    """Test the first conditional that source bounding box is inside observation
    bounding box.
    """
    checker = SceneValidator(scene=full_overlap_scene, parameters=None, observation=good_obs)
    results = checker.check_source_boxes_comparable_to_observation()

    assert isinstance(results[0], ValidationInfo)


def test_check_source_boxes_comparable_to_observation_no_overlap(no_overlap_scene, good_obs):
    """Test the second conditional that source bounding box is partially inside
    the observation bounding box.
    """
    checker = SceneValidator(scene=no_overlap_scene, parameters=None, observation=good_obs)
    results = checker.check_source_boxes_comparable_to_observation()

    assert isinstance(results[0], ValidationWarning)
    assert "outside" in results[0].message


def test_check_source_boxes_comparable_to_observation_partial_overlap(partial_overlap_scene, good_obs):
    """Test the third conditional that source bounding box is partially overlapping
    with observation bounding box.
    """
    checker = SceneValidator(scene=partial_overlap_scene, parameters=None, observation=good_obs)
    results = checker.check_source_boxes_comparable_to_observation()

    assert isinstance(results[0], ValidationWarning)
    assert "partially inside" in results[0].message
