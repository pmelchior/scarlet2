# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103
# ruff: noqa: D106

import copy

import astropy.units as u
import jax.numpy as jnp
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from scarlet2.frame import _flip_matrix, _rot_matrix
from scarlet2.measure import Moments
from scarlet2.morphology import GaussianMorphology

# create a test image and measure moments
t0 = 10
ellipticity = jnp.array((0.3, 0.5))
morph = GaussianMorphology(size=t0, ellipticity=ellipticity)
g = Moments(component=morph(), N=2)

# create trivial WCS for that image
wcs = WCS(naxis=2)
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
wcs.wcs.pc = jnp.diag(jnp.ones(2))


def test_measure_size():
    assert_allclose(t0, g.size, rtol=1e-3)


def test_measure_ellipticity():
    assert_allclose(ellipticity, g.ellipticity, rtol=2e-3)


def test_gaussian_from_moments():
    # generate Gaussian from moments
    t = g.size
    ellipticity = g.ellipticity
    morph2 = GaussianMorphology(t, ellipticity)

    assert_allclose(morph(), morph2(), rtol=1e-2)


def test_convolve_moments():
    g_ = copy.deepcopy(g)
    tp = t0 / 2
    psf = GaussianMorphology(size=tp, ellipticity=ellipticity)()
    psf /= psf.sum()
    p = Moments(psf, N=2)
    g_.convolve(p)
    assert_allclose(g_.size, jnp.sqrt(t0**2 + tp**2), rtol=3e-4)
    assert_allclose(g_.ellipticity, ellipticity, rtol=2e-3)
    g_.deconvolve(p)
    assert_allclose(g_.size, t0, rtol=3e-4)
    assert_allclose(g_.ellipticity, ellipticity, rtol=2e-3)


def test_rotate_moments():
    # rotate moments counter-clockwise 30 deg
    g_ = copy.deepcopy(g)
    a = 30 * u.deg
    g_.rotate(a)

    # apply theoretical rotation to spin-2 vector
    ellipticity_ = ellipticity[0] + 1j * ellipticity[1]
    ellipticity_ *= jnp.exp(2j * a.to(u.rad).value)
    ellipticity_ = jnp.array((ellipticity_.real, ellipticity_.imag))

    assert_allclose(g_.ellipticity, ellipticity_, rtol=2e-3)


def test_resize_moments():
    # resize the image
    c = 0.5
    morph2 = GaussianMorphology(size=t0 * c, ellipticity=ellipticity, shape=morph().shape)
    g2 = Moments(morph2(), 2)
    g2.resize(1 / c)

    assert_allclose(g.size, g2.size, rtol=1e-3)


def test_flip_moments():
    morph2 = jnp.fliplr(morph())
    g2 = Moments(morph2, 2)
    g2.resize((1, -1))  # flip second=x-axis

    assert_allclose(g.ellipticity, g2.ellipticity, rtol=1e-3)


def test_wcs_transfer_moments_rot90():
    # create a 90 deg counter-clockwise version of the morph image
    im_ = jnp.rot90(morph(), axes=(1, 0))
    g_ = Moments(im_)

    # create mock WCS for that image
    wcs_ = WCS(naxis=2)
    wcs_.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    phi = (-90 * u.deg).to(u.rad).value  # clockwise to counteract the rotation above
    wcs_.wcs.pc = _rot_matrix(phi)

    # match WCS
    g_.transfer(wcs_, wcs)

    # Check that size and ellipticity are conserved
    assert_allclose(g_.size, g.size, rtol=1e-3)
    assert_allclose(g_.ellipticity, g.ellipticity, rtol=1e-2)


def test_wcs_transfer_moments():
    # create a rotated, resized, flipped version of the morph image
    # apply theoretical rotation to spin-2 vector
    a = (30 * u.deg).to(u.rad).value
    ellipticity_ = ellipticity[0] + 1j * ellipticity[1]
    ellipticity_ *= jnp.exp(2j * a)
    ellipticity_ = jnp.array((ellipticity_.real, ellipticity_.imag))
    c = 0.5
    morph2 = GaussianMorphology(size=t0 * c, ellipticity=ellipticity_, shape=morph().shape)
    im_ = jnp.flipud(morph2())
    g_ = Moments(im_)

    # create mock WCS for that image
    wcs_ = WCS(naxis=2)
    wcs_.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_.wcs.pc = 1 / c * (_flip_matrix(-1) @ _rot_matrix(-a))

    # match WCS
    g_.transfer(wcs_, wcs)

    # Check that size and ellipticity are conserved
    assert_allclose(g_.size, g.size, rtol=1e-3)
    assert_allclose(g_.ellipticity, g.ellipticity, rtol=1e-2)
