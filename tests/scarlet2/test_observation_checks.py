import numpy as np
import pytest
from scarlet2.observation import Observation, ObservationValidator
from scarlet2.validation_utils import ValidationError, set_validation


@pytest.fixture(autouse=True)
def setup_validation():
    """Automatically enable validation for all tests."""
    set_validation(False)


@pytest.fixture()
def bad_obs():
    """Create an observation that should fail multiple validation checks."""
    return Observation(
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), weights=np.array([np.inf, 2.0, -1.0])
    )


@pytest.fixture()
def good_obs():
    """Create an observation that should pass all validation checks."""
    return Observation(
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        weights=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        channels=[0, 1],
    )


def test_weights_non_negative_returns_error(bad_obs):
    """Test that the weights in the observation are non-negative."""
    checker = ObservationValidator(bad_obs)

    results = checker.check_weights_non_negative()

    assert results is not None
    assert isinstance(results, ValidationError)


def test_weights_finite_returns_error(bad_obs):
    """Test that the weights in the observation are finite."""
    checker = ObservationValidator(bad_obs)

    results = checker.check_weights_finite()

    assert results is not None
    assert isinstance(results, ValidationError)


def test_weights_non_negative_returns_none(good_obs):
    """Test that the weights in the observation are non-negative."""
    checker = ObservationValidator(good_obs)

    results = checker.check_weights_non_negative()

    assert results is None


def test_weights_finite_returns_none(good_obs):
    """Test that the weights in the observation are finite."""
    checker = ObservationValidator(good_obs)

    results = checker.check_weights_finite()

    assert results is None


def test_data_and_weights_shape_returns_error(bad_obs):
    """Test that the data and weights have the same shape."""
    checker = ObservationValidator(bad_obs)

    results = checker.check_data_and_weights_shape()

    assert results is not None
    assert isinstance(results, ValidationError)
    assert results.message == "Data and weights must have the same shape."


def test_data_and_weights_shape_returns_none(good_obs):
    """Test that the data and weights have the same shape."""
    checker = ObservationValidator(good_obs)

    results = checker.check_data_and_weights_shape()

    assert results is None


def test_num_channels_matches_data_returns_none(good_obs):
    """Test that the number of channels in the observation matches the data."""
    checker = ObservationValidator(good_obs)

    results = checker.check_num_channels_matches_data()

    assert results is None


def test_data_finite_for_non_zero_weights_returns_none(good_obs):
    """Test that the data in the observation is finite where weights are greater than zero."""
    checker = ObservationValidator(good_obs)

    results = checker.check_data_finite_for_non_zero_weights()

    assert results is None


def test_data_finite_for_non_zero_weights_returns_none_with_infinity():
    """Test that non-finite data does not raise an error when weights are zero."""
    obs = Observation(
        data=np.array([[np.inf, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        weights=np.array([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        channels=[0, 1],
    )

    checker = ObservationValidator(obs)
    results = checker.check_data_finite_for_non_zero_weights()

    assert results is None


def test_data_finite_for_non_zero_weights_returns_error_with_infinity():
    """Test that non-finite data raises an error when weights are non-zero."""
    obs = Observation(
        data=np.array([[np.inf, np.inf, 3.0], [4.0, 5.0, 6.0]]),
        weights=np.array([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        channels=[0, 1],
    )

    checker = ObservationValidator(obs)
    results = checker.check_data_finite_for_non_zero_weights()

    assert results is not None
    assert isinstance(results, ValidationError)
    assert results.message == "Data in the observation must be finite."
