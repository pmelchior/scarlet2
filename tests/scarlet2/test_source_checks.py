import pytest
from scarlet2.source import SourceValidator
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
