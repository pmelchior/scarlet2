from unittest.mock import patch

import pytest
from scarlet2.scene import FitValidator
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
