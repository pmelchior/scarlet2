import jax.numpy as jnp
from jax import vmap
import scarlet2
from scarlet2 import *
from numpy.testing import assert_allclose
import astropy.units as u

from scarlet2.measure import get_scale
from scarlet_test_data import data_path
import os
import astropy.io.fits as fits
from astropy.wcs import WCS

T0 = 30
ellipticity = jnp.array((0.3,0.5))
morph = GaussianMorphology(size=T0, ellipticity=ellipticity)

print("Constructed moments")
print(f"Size: {T0}, Ellipticity: {ellipticity}")

g = scarlet2.measure.moments(component=morph(), N=2)

def test_measure_size():
    g = scarlet2.measure.moments(component=morph(), N=2)
    print("Measured moments")
    print(f"Size: {g.size}")
    assert_allclose(T0, g.size, rtol=1e-3)

def test_measure_ellipticity():
    g = scarlet2.measure.moments(component=morph(), N=2)
    print("Measured moments")
    print(f"Ellipticity: {g.ellipticity}")
    assert_allclose(ellipticity, g.ellipticity, rtol=2e-3)

def test_gaussian_from_moments():
    g = scarlet2.measure.moments(component=morph(), N=2)
    # generate Gaussian from moments
    T = g.size
    ellipticity = g.ellipticity
    morph2 = GaussianMorphology(T, ellipticity)
    assert_allclose(morph(), morph2(), rtol=1e-2)

def test_rotate_moments():
    a = 90*u.deg
    a.to(u.deg).value
    # rotate 90 degrees counterclockwise the image
    # and measure its moments
    g2 = scarlet2.measure.moments(jnp.rot90(morph()))

    # rotate moments computed on the original image
    g.rotate(a)
    g.ellipticity

    assert_allclose(g.ellipticity, g2.ellipticity, rtol=1e-6)

def test_resize_moments():
    c = 0.5

    # resize the image
    morph2 = GaussianMorphology(size=T0*c, 
                            ellipticity=ellipticity,
                            shape=morph().shape)
    g2 = scarlet2.measure.moments(morph2(),2)

    # resize moments computed on the original image
    g = scarlet2.measure.moments(component=morph(), N=2)
    g.resize(0.5)

    assert_allclose(g.size, g2.size, rtol=1e-3)

def test_wcs_transfer_moments():

    # Load the HSC image WCS
    obs_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HSC1.fits"))
    wcs_hsc = WCS(obs_hdu[0].header)

    # Load the HST image WCS
    hst_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HST1.fits"))
    wcs_hst = WCS(hst_hdu[0].header)

    # Mock a rotation of 90 deg counter-clockwise of the HST WCS 
    phi = 90 / 180 * jnp.pi # in rad
    R = jnp.array([[jnp.cos(phi), jnp.sin(phi)],
                   [-jnp.sin(phi), jnp.cos(phi)]])

    wcs_hst.wcs.pc = R @ wcs_hst.wcs.pc
    im_hst = jnp.rot90(morph())

    # Generate the same image seen from HSC
    h = (get_scale(wcs_hst) / get_scale(wcs_hsc)).mean()
    T1 = T0 * h 
    ellipticity1 = jnp.array((0.3,0.5))
    morph1 = GaussianMorphology(size=T1, ellipticity=ellipticity1, shape=im_hst.shape)
    im_hsc = morph1()

    # Measure moments of the HST image
    g0 = scarlet2.measure.moments(im_hst)

    # Measure moments of the HSC image
    g1 = scarlet2.measure.moments(im_hsc)

    # Transfer moments from HST to HSC frame
    g0.transfer(wcs_hst, wcs_hsc)

    # Check that size and ellipticity are conserved
    assert_allclose(g1.size, g0.size, rtol=1e-3)
    assert_allclose(g1.ellipticity, g0.ellipticity, rtol=1e-2)

def test_wcs_transfer_w_flip_moments():

    # same as above with RA W->E instead of E->W convention in HST WCS
    # Load the HSC image WCS
    obs_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HSC1.fits"))
    wcs_hsc = WCS(obs_hdu[0].header)

    # Load the HST image WCS
    hst_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HST1.fits"))
    wcs_hst = WCS(hst_hdu[0].header)

    # Mock a rotation of 90 deg counter-clockwise of the HST WCS 
    phi = 90 / 180 * jnp.pi # in rad
    R = jnp.array([[jnp.cos(phi), jnp.sin(phi)],
                   [-jnp.sin(phi), jnp.cos(phi)]])

    wcs_hst.wcs.pc = R @ wcs_hst.wcs.pc
    wcs_hst.wcs.pc *= jnp.array([[-1], [1]])
    
    # rotate and flip x
    im_hst = jnp.rot90(morph())[:,::-1]

    # Generate the same image seen from HSC
    h = (get_scale(wcs_hst) / get_scale(wcs_hsc)).mean()
    T1 = T0 * h 
    ellipticity1 = jnp.array((0.3,0.5))
    morph1 = GaussianMorphology(size=T1, ellipticity=ellipticity1, shape=im_hst.shape)
    im_hsc = morph1()

    # Measure moments of the HST image
    g0 = scarlet2.measure.moments(im_hst)

    # Measure moments of the HSC image
    g1 = scarlet2.measure.moments(im_hsc)

    # Transfer moments from HST to HSC frame
    g0.transfer(wcs_hst, wcs_hsc)

    # Check that size and ellipticity are conserved
    assert_allclose(g1.size, g0.size, rtol=1e-3)
    assert_allclose(g1.ellipticity, g0.ellipticity, rtol=1e-2)

def test_wcs_transfer_moments_multichannels():

    # Load the HSC image WCS
    obs_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HSC1.fits"))
    wcs_hsc = WCS(obs_hdu[0].header)

    # Load the HST image WCS
    hst_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HST1.fits"))
    wcs_hst = WCS(hst_hdu[0].header)

    # Mock a rotation of 90 deg counter-clockwise of the HST WCS 
    phi = 90 / 180 * jnp.pi # in rad
    R = jnp.array([[jnp.cos(phi), jnp.sin(phi)],
                   [-jnp.sin(phi), jnp.cos(phi)]])

    wcs_hst.wcs.pc = R @ wcs_hst.wcs.pc

    nc = 5
    im_hst = jnp.repeat(morph()[None,:,:], repeats=nc, axis=0)
    im_hst = vmap(jnp.rot90)(im_hst)

    # Generate the same image seen from HSC
    h = (get_scale(wcs_hst) / get_scale(wcs_hsc)).mean()
    T1 = T0 * h 
    ellipticity1 = jnp.array((0.3,0.5))
    morph1 = GaussianMorphology(size=T1, ellipticity=ellipticity1, shape=im_hst.shape)
    im_hsc = morph1()

    # Measure moments of the HST image
    g0 = scarlet2.measure.moments(im_hst)

    # Measure moments of the HSC image
    g1 = scarlet2.measure.moments(im_hsc)

    # Transfer moments from HST to HSC frame
    g0.transfer(wcs_hst, wcs_hsc)
    # Check that size and ellipticity are conserved
    assert_allclose(jnp.repeat(g1.size, nc), g0.size, rtol=1e-3)
    assert_allclose(jnp.repeat(g1.ellipticity[:,None], nc, 1), g0.ellipticity, rtol=1e-2)