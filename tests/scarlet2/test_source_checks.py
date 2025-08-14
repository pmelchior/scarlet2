import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from scarlet2 import init
from scarlet2.frame import Frame
from scarlet2.observation import Observation
from scarlet2.psf import ArrayPSF
from scarlet2.scene import Scene
from scarlet2.source import Source, SourceValidator
from scarlet2.validation_utils import (
    ValidationError,
    ValidationInfo,
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
def obs(data_file):
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
def good_source(obs, data_file):
    """Assemble a source from the good observation and the data file."""
    model_frame = Frame.from_observations(obs)
    with Scene(model_frame) as _:
        centers = np.array([(src["y"], src["x"]) for src in data_file["catalog"]])  # Note: y/x convention!
        spectrum, morph = init.from_gaussian_moments(obs, centers[0], min_corr=0.99)
        return Source(centers[0], spectrum, morph)


@pytest.fixture()
def bad_source(obs, data_file):
    """Assemble a source from the bad observation and the data file."""
    model_frame = Frame.from_observations(obs)
    with Scene(model_frame) as _:
        centers = np.array([(src["y"], src["x"]) for src in data_file["catalog"]])  # Note: y/x convention!
        spectrum, morph = init.from_gaussian_moments(obs, centers[0], min_corr=0.99)
        return Source(centers[0], spectrum * -1, morph)


def test_check_source_has_positive_contribution(good_source):
    """Check that the source has a positive contribution to the observation."""
    checker = SourceValidator(good_source)
    results = checker.check_source_has_positive_contribution()

    assert isinstance(results, ValidationInfo)


def test_check_source_has_positive_contribution_fails_with_negative_values(bad_source):
    """Expect a ValidationError when source has a negative contribution."""
    checker = SourceValidator(bad_source)
    results = checker.check_source_has_positive_contribution()

    assert isinstance(results, ValidationError)
    assert results.message == "Source model has negative contributions."
