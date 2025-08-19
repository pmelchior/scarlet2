import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.distributions import constraints
from scarlet2 import init
from scarlet2.frame import Frame
from scarlet2.module import Parameter
from scarlet2.scene import Scene, SceneValidator
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
def full_overlap_scene(good_obs, data_file):
    """Assemble a scene from the good observation and the data file."""
    model_frame = Frame.from_observations(good_obs)
    centers = jnp.array([(src["y"], src["x"]) for src in data_file["catalog"]])  # Note: y/x convention!

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
def bad_parameters(scene, good_obs):
    """Create faulty parameters for a scene. Namely 1) stepsize=None and no prior
    is defined. 2) stepsize and prior both explicitly = None."""

    parameters = scene.make_parameters()
    for i in range(len(scene.sources)):
        parameters += Parameter(
            scene.sources[i].spectrum,
            name=f"spectrum:{i}",
            constraint=constraints.positive,
            stepsize=None,
        )
        if i == 0:
            parameters += Parameter(scene.sources[i].center, name=f"center:{i}", stepsize=0.1)
        else:
            parameters += Parameter(
                scene.sources[i].morphology,
                name=f"morph:{i}",
                constraint=constraints.unit_interval,
                stepsize=None,
                prior=None,
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


def test_check_parameters_have_necessary_fields(parameters):
    """Test that all parameters have the necessary fields set."""
    checker = SceneValidator(scene=None, parameters=parameters, observation=None)
    results = checker.check_parameters_have_necessary_fields()

    assert isinstance(results[0], ValidationInfo)


def test_check_parameters_have_necessary_fields_both_prior_and_stepsize(bad_parameters):
    """Test that all parameters have the necessary fields set."""
    checker = SceneValidator(scene=None, parameters=bad_parameters, observation=None)
    results = checker.check_parameters_have_necessary_fields()

    assert isinstance(results[0], ValidationError)
    assert "does not have prior or stepsize" in results[0].message
