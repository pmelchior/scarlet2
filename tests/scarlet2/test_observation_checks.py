import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from scarlet2.observation import Observation, ObservationValidator
from scarlet2.psf import ArrayPSF
from scarlet2.validation_utils import ValidationError, ValidationInfo, set_validation


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


def test_weights_non_negative_returns_error(bad_obs):
    """Test that the weights in the observation are non-negative."""
    checker = ObservationValidator(bad_obs)

    results = checker.check_weights_non_negative()

    assert isinstance(results, ValidationError)


def test_weights_finite_returns_error(bad_obs):
    """Test that the weights in the observation are finite."""
    checker = ObservationValidator(bad_obs)

    results = checker.check_weights_finite()

    assert isinstance(results, ValidationError)


def test_weights_non_negative_returns_none(good_obs):
    """Test that the weights in the observation are non-negative."""
    checker = ObservationValidator(good_obs)

    results = checker.check_weights_non_negative()

    assert isinstance(results, ValidationInfo)


def test_weights_finite_returns_none(good_obs):
    """Test that the weights in the observation are finite."""
    checker = ObservationValidator(good_obs)

    results = checker.check_weights_finite()

    assert isinstance(results, ValidationInfo)


def test_data_and_weights_shape_returns_error(bad_obs):
    """Test that the data and weights have the same shape."""
    checker = ObservationValidator(bad_obs)

    results = checker.check_data_and_weights_shape()

    assert isinstance(results, ValidationError)
    assert results.message == "Data and weights must have the same shape."


def test_data_and_weights_shape_returns_none(good_obs):
    """Test that the data and weights have the same shape."""
    checker = ObservationValidator(good_obs)

    results = checker.check_data_and_weights_shape()

    assert isinstance(results, ValidationInfo)


def test_num_channels_matches_data_returns_none(good_obs):
    """Test that the number of channels in the observation matches the data."""
    checker = ObservationValidator(good_obs)

    results = checker.check_num_channels_matches_data()

    assert isinstance(results, ValidationInfo)


def test_data_finite_for_non_zero_weights_returns_none(good_obs):
    """Test that the data in the observation is finite where weights are greater than zero."""
    checker = ObservationValidator(good_obs)

    results = checker.check_data_finite_for_non_zero_weights()

    assert isinstance(results, ValidationInfo)


def test_data_finite_for_non_zero_weights_returns_none_with_infinity():
    """Test that non-finite data does not raise an error when weights are zero."""
    obs = Observation(
        data=np.array([[np.inf, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        weights=np.array([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        channels=[0, 1],
    )

    checker = ObservationValidator(obs)
    results = checker.check_data_finite_for_non_zero_weights()

    assert isinstance(results, ValidationInfo)


def test_data_finite_for_non_zero_weights_returns_error_with_infinity():
    """Test that non-finite data raises an error when weights are non-zero."""
    obs = Observation(
        data=np.array([[np.inf, np.inf, 3.0], [4.0, 5.0, 6.0]]),
        weights=np.array([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        channels=[0, 1],
    )

    checker = ObservationValidator(obs)
    results = checker.check_data_finite_for_non_zero_weights()

    assert isinstance(results, ValidationError)
    assert results.message == "Data in the observation must be finite."


def test_number_of_psf_channels(good_obs):
    """Test that the number of PSF channels matches the observation channels."""
    checker = ObservationValidator(good_obs)

    results = checker.check_number_of_psf_channels()
    assert isinstance(results, ValidationInfo)


def test_number_of_psf_channels_returns_error(bad_obs):
    """Test that the number of PSF channels does not match the observation channels."""
    checker = ObservationValidator(bad_obs)

    results = checker.check_psf_has_3_dimensions()

    assert isinstance(results, ValidationError)
    assert results.message == "PSF must be 3-dimensional."


def test_validation_on_runs_observation_checks(capsys):
    """Set auto-validation to True, expect that the non-finite data raises an error
    when weights are non-zero."""
    set_validation(True)
    _ = Observation(
        data=np.array([[np.inf, np.inf, 3.0], [4.0, 5.0, 6.0]]),
        weights=np.array([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        channels=[0, 1],
    )
    captured = capsys.readouterr()
    assert "Observation validation results:" in captured.out


def test_psf_centroid_inconsistent_returns_error(data_file):
    """Test that when the PSF centroids are not consistent across bands, an error
    is raised."""

    data = np.asarray(data_file["images"])
    channels = [str(f) for f in data_file["filters"]]
    weights = np.asarray(1 / data_file["variance"])
    psf = np.asarray(data_file["psfs"])
    psf[0][1:, :] = psf[0][:-1, :]  # Shift the PSF to create an inconsistency

    bad_obs = Observation(
        data=data,
        weights=weights,
        channels=channels,
        psf=ArrayPSF(psf),
    )

    checker = ObservationValidator(bad_obs)
    results = checker.check_psf_centroid_consistent()

    assert isinstance(results, ValidationError)
    assert results.message == "PSF centroid is not the same in all channels."


def test_psf_centroid_consistent_no_error(data_file):
    """Test that when the PSF centroids are consistent across bands, no errors
    are raised."""

    data = np.asarray(data_file["images"])
    channels = [str(f) for f in data_file["filters"]]
    weights = np.asarray(1 / data_file["variance"])
    psf = np.zeros_like(data)
    psf[:, 1, 1] = 1

    bad_obs = Observation(
        data=data,
        weights=weights,
        channels=channels,
        psf=ArrayPSF(psf),
    )

    checker = ObservationValidator(bad_obs)

    results = checker.check_psf_centroid_consistent()
    assert isinstance(results, ValidationInfo)
