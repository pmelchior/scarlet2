from unittest.mock import patch

import pytest

from scarlet2.infer import FitValidator
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
def test_check_goodness_of_fit(scene, good_obs, mocked_chi_value, expected):
    """Mocked goodness_of_fit return to produces the expected ValidationResult type"""

    scene_ = scene.fit(good_obs, max_iter=1, e_rel=1e-4, progress_bar=False)
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
def test_check_chi_squared_in_box_and_border(scene, good_obs, mocked_chi_value, expected):
    """Mocked chi-squared evaluation in box and border."""

    scene_ = scene.fit(good_obs, max_iter=1, e_rel=1e-4, progress_bar=False)
    checker = FitValidator(scene_, good_obs)
    mock_return = {0: {"in": mocked_chi_value, "out": mocked_chi_value}}

    with patch.object(type(good_obs), "eval_chi_square_in_box_and_border", return_value=mock_return) as _:
        results = checker.check_chi_square_in_box_and_border()

    assert all(isinstance(res, expected) for res in results)


def _chi_square_reference(residuals, weights, bbox, border_width):
    """Brute-force chi-square in box and border with explicit pixel masks"""
    import numpy as np

    from scarlet2.bbox import Box

    chi2 = np.asarray(weights * residuals**2)
    inside = np.zeros(chi2.shape, dtype=bool)
    inside[bbox.slices] = True
    outside = np.zeros(chi2.shape, dtype=bool)
    # clip to image: negative slice starts would wrap around
    outside[(bbox.grow([0, border_width, border_width]) & Box(chi2.shape)).slices] = True
    outside &= ~inside
    return chi2[inside].mean(), chi2[outside].mean()


@pytest.mark.parametrize(
    "shape, origin, border_width",
    [
        ((2, 6, 6), (0, 5, 5), 3),  # interior
        ((2, 6, 6), (0, 5, 5), 2),  # border width other than 3
        ((2, 6, 6), (0, 0, 0), 3),  # touching the lower edges
        ((2, 6, 8), (0, 14, 12), 3),  # touching the upper edges
        ((2, 18, 10), (0, 1, 2), 3),  # border only on some sides
    ],
)
def test_chi_square_in_box_and_border(shape, origin, border_width):
    """Chi-square in box and border, including boxes whose border extends beyond the image"""
    import jax
    import numpy as np

    from scarlet2.bbox import Box
    from scarlet2.observation import chi_square_in_box_and_border

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    residuals = jax.random.normal(key1, (2, 20, 20))
    weights = jax.random.uniform(key2, (2, 20, 20))
    bbox = Box(shape, origin=origin)

    chi_in, chi_out = chi_square_in_box_and_border(residuals, weights, bbox, border_width)
    ref_in, ref_out = _chi_square_reference(residuals, weights, bbox, border_width)
    np.testing.assert_allclose(chi_in, ref_in, rtol=1e-6)
    np.testing.assert_allclose(chi_out, ref_out, rtol=1e-6)


def test_chi_square_box_covers_image():
    """Without border pixels, the border chi-square is nan and produces no validation result"""
    import jax.numpy as jnp

    from scarlet2.bbox import Box
    from scarlet2.observation import chi_square_in_box_and_border

    residuals = jnp.ones((2, 10, 10))
    chi_in, chi_out = chi_square_in_box_and_border(residuals, jnp.ones_like(residuals), Box((2, 10, 10)), 3)
    assert chi_in == 1
    assert jnp.isnan(chi_out)
