import jax.numpy as jnp
import scarlet2
from scarlet2 import *
from numpy.testing import assert_allclose
import astropy.units as u

T0 = 30
center = jnp.array([10, 10])
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


