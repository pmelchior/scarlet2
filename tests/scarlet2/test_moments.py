# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103
# ruff: noqa: D106

import astropy.io.fits as fits
import astropy.units as u
import jax.numpy as jnp
import scarlet2
from astropy.wcs import WCS
from huggingface_hub import hf_hub_download
from jax import vmap
from numpy.testing import assert_allclose
from scarlet2 import *  # noqa: F403
from scarlet2.measure import get_scale
from scarlet2.morphology import GaussianMorphology

# load test data from HSC and HST
filename_hsc = hf_hub_download(
    repo_id="astro-data-lab/scarlet-test-data",
    filename="test_resampling/Cut_HSC1.fits.gz",
    repo_type="dataset",
)
hsc_hdu = fits.open(filename_hsc)
wcs_hsc = WCS(hsc_hdu[0].header)

filename_hst = hf_hub_download(
    repo_id="astro-data-lab/scarlet-test-data",
    filename="test_resampling/Cut_HST1.fits.gz",
    repo_type="dataset",
)
hst_hdu = fits.open(filename_hst)
wcs_hst = WCS(hst_hdu[0].header)

T0 = 30
ellipticity = jnp.array((0.3, 0.5))
morph = GaussianMorphology(size=T0, ellipticity=ellipticity)

print("Constructed moments")
print(f"Size: {T0}, Ellipticity: {ellipticity}")

g = scarlet2.measure.Moments(component=morph(), N=2)


def test_measure_size():
    g = scarlet2.measure.Moments(component=morph(), N=2)
    print("Measured moments")
    print(f"Size: {g.size}")
    assert_allclose(T0, g.size, rtol=1e-3)


def test_measure_ellipticity():
    g = scarlet2.measure.Moments(component=morph(), N=2)
    print("Measured moments")
    print(f"Ellipticity: {g.ellipticity}")
    assert_allclose(ellipticity, g.ellipticity, rtol=2e-3)


def test_gaussian_from_moments():
    g = scarlet2.measure.Moments(component=morph(), N=2)
    # generate Gaussian from moments
    t = g.size
    ellipticity = g.ellipticity
    morph2 = GaussianMorphology(t, ellipticity)
    assert_allclose(morph(), morph2(), rtol=1e-2)


def test_rotate_moments():
    a = 90 * u.deg
    a.to(u.deg).value  # noqa: B018
    # rotate 90 degrees counterclockwise the image
    # and measure its moments
    g2 = scarlet2.measure.Moments(jnp.rot90(morph()))

    # rotate moments computed on the original image
    g.rotate(a)

    assert_allclose(g.ellipticity, g2.ellipticity, rtol=1e-6)


def test_resize_moments():
    c = 0.5

    # resize the image
    morph2 = GaussianMorphology(size=T0 * c, ellipticity=ellipticity, shape=morph().shape)
    g2 = scarlet2.measure.Moments(morph2(), 2)

    # resize moments computed on the original image
    g = scarlet2.measure.Moments(component=morph(), N=2)
    g.resize(0.5)

    assert_allclose(g.size, g2.size, rtol=1e-3)


def test_wcs_transfer_moments():
    # Mock a rotation of 90 deg counter-clockwise of the HST WCS
    phi = 90 / 180 * jnp.pi  # in rad
    r = jnp.array([[jnp.cos(phi), jnp.sin(phi)], [-jnp.sin(phi), jnp.cos(phi)]])

    wcs_hst = WCS(hst_hdu[0].header)  # need to recreate to change in the next step
    wcs_hst.wcs.pc = r @ wcs_hst.wcs.pc
    im_hst = jnp.rot90(morph())

    # Generate the same image seen from HSC
    h = (get_scale(wcs_hst) / get_scale(wcs_hsc)).mean()
    t1 = T0 * h
    ellipticity1 = jnp.array((0.3, 0.5))
    morph1 = GaussianMorphology(size=t1, ellipticity=ellipticity1, shape=im_hst.shape)
    im_hsc = morph1()

    # Measure moments of the HST image
    g0 = scarlet2.measure.Moments(im_hst)

    # Measure moments of the HSC image
    g1 = scarlet2.measure.Moments(im_hsc)

    # Transfer moments from HST to HSC frame
    g0.transfer(wcs_hst, wcs_hsc)

    # Check that size and ellipticity are conserved
    assert_allclose(g1.size, g0.size, rtol=1e-3)
    assert_allclose(g1.ellipticity, g0.ellipticity, rtol=1e-2)


def test_wcs_transfer_w_flip_moments():
    # Mock a rotation of 90 deg counter-clockwise of the HST WCS
    phi = 90 / 180 * jnp.pi  # in rad
    r = jnp.array([[jnp.cos(phi), jnp.sin(phi)], [-jnp.sin(phi), jnp.cos(phi)]])

    wcs_hst = WCS(hst_hdu[0].header)  # need to recreate to change in the next step
    wcs_hst.wcs.pc = r @ wcs_hst.wcs.pc
    wcs_hst.wcs.pc *= jnp.array([[-1], [1]])

    # rotate and flip x
    im_hst = jnp.rot90(morph())[:, ::-1]

    # Generate the same image seen from HSC
    h = (get_scale(wcs_hst) / get_scale(wcs_hsc)).mean()
    t1 = T0 * h
    ellipticity1 = jnp.array((0.3, 0.5))
    morph1 = GaussianMorphology(size=t1, ellipticity=ellipticity1, shape=im_hst.shape)
    im_hsc = morph1()

    # Measure moments of the HST image
    g0 = scarlet2.measure.Moments(im_hst)

    # Measure moments of the HSC image
    g1 = scarlet2.measure.Moments(im_hsc)

    # Transfer moments from HST to HSC frame
    g0.transfer(wcs_hst, wcs_hsc)

    # Check that size and ellipticity are conserved
    assert_allclose(g1.size, g0.size, rtol=1e-3)
    assert_allclose(g1.ellipticity, g0.ellipticity, rtol=1e-2)


def test_wcs_transfer_moments_multichannels():
    # Mock a rotation of 90 deg counter-clockwise of the HST WCS
    phi = 90 / 180 * jnp.pi  # in rad
    r = jnp.array([[jnp.cos(phi), jnp.sin(phi)], [-jnp.sin(phi), jnp.cos(phi)]])

    wcs_hst = WCS(hst_hdu[0].header)  # need to recreate to change in the next step
    wcs_hst.wcs.pc = r @ wcs_hst.wcs.pc

    nc = 5
    im_hst = jnp.repeat(morph()[None, :, :], repeats=nc, axis=0)
    im_hst = vmap(jnp.rot90)(im_hst)

    # Generate the same image seen from HSC
    h = (get_scale(wcs_hst) / get_scale(wcs_hsc)).mean()
    t1 = T0 * h
    ellipticity1 = jnp.array((0.3, 0.5))
    morph1 = GaussianMorphology(size=t1, ellipticity=ellipticity1, shape=im_hst.shape)
    im_hsc = morph1()

    # Measure moments of the HST image
    g0 = scarlet2.measure.Moments(im_hst)

    # Measure moments of the HSC image
    g1 = scarlet2.measure.Moments(im_hsc)

    # Transfer moments from HST to HSC frame
    g0.transfer(wcs_hst, wcs_hsc)
    # Check that size and ellipticity are conserved
    assert_allclose(jnp.repeat(g1.size, nc), g0.size, rtol=1e-3)
    assert_allclose(jnp.repeat(g1.ellipticity[:, None], nc, 1), g0.ellipticity, rtol=1e-2)
