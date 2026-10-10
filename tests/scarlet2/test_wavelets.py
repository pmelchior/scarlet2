# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scarlet2.morphology import StarletMorphology
from scarlet2.wavelets import starlet_filter_norms, starlet_reconstruction, starlet_transform


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    y, x = np.mgrid[-20:21, -20:21]
    return jnp.asarray(np.exp(-(x**2 + y**2) / 18) + 0.1 * rng.normal(size=x.shape), dtype=jnp.float32)


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize("scales", [1, 2, 3, 4])
def test_starlet_roundtrip(image, generation, scales):
    coeffs = starlet_transform(image, scales=scales, generation=generation)
    recon = starlet_reconstruction(coeffs, generation=generation)
    np.testing.assert_allclose(recon, image, atol=1e-5)


def test_starlet_morphology_roundtrip(image):
    morph = StarletMorphology.from_image(image)
    np.testing.assert_allclose(morph(), image, atol=1e-5)


def test_starlet_reconstruction_subset(image):
    coeffs = starlet_transform(image)
    max_scale = len(coeffs) - 1
    # reconstruction is linear in the coefficients: disjoint subsets add up to the full image
    fine = starlet_reconstruction(coeffs, scales=[0, 1])
    coarse = starlet_reconstruction(coeffs, scales=range(2, max_scale + 1))
    np.testing.assert_allclose(fine + coarse, image, atol=1e-5)


@pytest.mark.parametrize("generation", [1, 2])
def test_starlet_filter_norms(generation):
    scales = 3
    norms = starlet_filter_norms(scales, generation=generation)
    assert norms.shape == (scales + 1,)
    # norms predict the per-level std of the transform of white noise
    noise = jax.random.normal(jax.random.key(0), (512, 512))
    coeffs = starlet_transform(noise, scales=scales, generation=generation)
    std = coeffs[:, 64:-64, 64:-64].std(axis=(-2, -1))
    np.testing.assert_allclose(std, norms, rtol=0.05)
    # subset of levels
    np.testing.assert_allclose(
        starlet_filter_norms(scales, levels=[0, 2], generation=generation), norms[jnp.array([0, 2])]
    )
