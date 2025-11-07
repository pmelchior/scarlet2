# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103
# ruff: noqa: D106

import astropy.units as u
import jax.numpy as jnp
from numpy.testing import assert_allclose

import scarlet2
from scarlet2 import ArrayPSF
from scarlet2.frame import _flip_matrix, _rot_matrix, _wcs_default, get_affine
from scarlet2.measure import Moments
from scarlet2.morphology import GaussianMorphology

# resampling renderer to test
cls = scarlet2.renderer.ResamplingRenderer

# create a Gaussian as model
T = 10
eps = jnp.array((0.5, 0.3))
model = GaussianMorphology(size=T, ellipticity=eps, shape=(150, 151))()[None, ...]  # test even & odd
g = scarlet2.measure.Moments(component=model[0], N=2)

# make a Frame
model_frame = scarlet2.Frame(scarlet2.Box(model.shape))


def test_rescale():
    # how does this model look if we change the WCS scale
    # if scale becomes larger, the image gets smaller
    scale = 3.1
    shape = (int(model.shape[-2] / scale) + 1, int(model.shape[-1] / scale))
    wcs_obs = _wcs_default(shape)
    m = scarlet2.frame.get_affine(wcs_obs)
    m = scale * m
    wcs_obs.wcs.pc = m

    obs_frame = scarlet2.Frame(scarlet2.Box(shape), wcs=wcs_obs)
    renderer = cls(model_frame, obs_frame)
    model_ = renderer(model)

    # for testing outputs...
    # print(renderer)
    # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # ax[0].imshow(model[0])
    # ax[1].imshow(model_[0])

    # undo resizing
    g_obs = Moments(model_[0])
    g_obs.resize(scale)
    assert_allclose(g_obs.flux, g.flux, atol=3e-3)
    shift_ = jnp.asarray(g_obs.centroid) - jnp.asarray(obs_frame.bbox.spatial.center)
    assert_allclose(shift_, (0, 0), atol=1e-4)
    assert_allclose(g_obs.size, g.size, rtol=3e-5)
    assert_allclose(g_obs.ellipticity, g.ellipticity, atol=3e-5)


def test_rotate():
    # how does this model look if we rotated the WCS scale
    # because we change the frame, this will appear as a rotation in the opposite direction
    shape = model.shape
    phi = (30 * u.deg).to(u.rad).value
    wcs_obs = _wcs_default(shape)
    m = scarlet2.frame.get_affine(wcs_obs)
    r = _rot_matrix(phi)
    m = r @ m
    wcs_obs.wcs.pc = m

    obs_frame = scarlet2.Frame(scarlet2.Box(shape), wcs=wcs_obs)
    renderer = cls(model_frame, obs_frame)
    model_ = renderer(model)

    # rotate to correct for the counter-rotation of the frame
    g_obs = Moments(model_[0])
    g_obs.rotate(phi)
    assert_allclose(g_obs.flux, g.flux, atol=3e-3)
    assert_allclose(g_obs.centroid, g.centroid, atol=1e-4)
    assert_allclose(g_obs.size, g.size, rtol=3e-5)
    assert_allclose(g_obs.ellipticity, g.ellipticity, atol=3e-5)


def test_flip():
    shape = model.shape
    wcs_obs = _wcs_default(shape)
    m = scarlet2.frame.get_affine(wcs_obs)
    f = _flip_matrix(-1)
    m = f @ m
    wcs_obs.wcs.pc = m

    obs_frame = scarlet2.Frame(scarlet2.Box(shape), wcs=wcs_obs)
    renderer = cls(model_frame, obs_frame)
    model_ = renderer(model)

    # undo the flip
    g_obs = Moments(model_[0])
    g_obs.flipud()
    assert_allclose(g_obs.flux, g.flux, atol=3e-3)
    assert_allclose(g_obs.centroid, g.centroid, atol=1e-4)
    assert_allclose(g_obs.size, g.size, rtol=3e-5)
    assert_allclose(g_obs.ellipticity, g.ellipticity, atol=3e-5)


def test_translation():
    shift = jnp.array((12, -1.9))
    shape = model.shape
    wcs_obs = _wcs_default(shape)
    wcs_obs.wcs.crpix += shift[::-1]  # x/y

    shape = model.shape
    obs_frame = scarlet2.Frame(scarlet2.Box(shape), wcs=wcs_obs)
    renderer = cls(model_frame, obs_frame)
    model_ = renderer(model)

    g_obs = Moments(model_[0])
    shift_ = jnp.asarray(g_obs.centroid) - jnp.asarray(g.centroid)

    assert_allclose(g_obs.flux, g.flux, rtol=2e-3)
    assert_allclose(shift_, -shift, atol=1e-4)
    assert_allclose(g_obs.size, g.size, rtol=3e-5)
    assert_allclose(g_obs.ellipticity, g.ellipticity, atol=3e-5)


def test_convolution():
    # create model PSF and convolve g
    t_p = 0.7
    eps_p = None
    psf = GaussianMorphology(size=t_p, ellipticity=eps_p)()
    psf /= psf.sum()
    p = Moments(psf, N=2)
    psf = ArrayPSF(psf[None, ...])
    model_frame_ = scarlet2.Frame(scarlet2.Box(model.shape), psf=psf)

    # create obs PSF
    t_obs, eps_obs = 5, jnp.array((-0.1, 0.1))
    psf_obs = GaussianMorphology(size=t_obs, ellipticity=eps_obs)()
    psf_obs /= psf_obs.sum()
    p_obs = Moments(psf_obs, N=2)
    psf_obs = ArrayPSF(psf_obs[None, ...])

    # render: deconvolve, reconcolve
    shape = model.shape
    obs_frame = scarlet2.Frame(scarlet2.Box(shape), psf=psf_obs)
    renderer = cls(model_frame_, obs_frame)
    model_ = renderer(model)
    g_obs = Moments(model_[0])

    g_obs.deconvolve(p_obs).convolve(p)
    assert_allclose(g_obs.flux, g.flux, atol=3e-3)
    assert_allclose(g_obs.centroid, g.centroid, atol=1e-4)
    assert_allclose(g_obs.size, g.size, rtol=3e-5)
    assert_allclose(g_obs.ellipticity, g.ellipticity, atol=3e-5)


def test_all():
    scale = 2.1
    phi = (70 * u.deg).to(u.rad).value
    shift = jnp.array((1.4, -0.456))
    shape = (int(model.shape[1] // scale), int(model.shape[2] // scale))
    wcs_obs = _wcs_default(shape)
    m = get_affine(wcs_obs)
    wcs_obs.wcs.pc = scale * _rot_matrix(phi) @ _flip_matrix(-1) @ m
    wcs_obs.wcs.crpix += shift[::-1]  # x/y

    # create model PSF and convolve g
    t_p = 1
    eps_p = None
    psf = GaussianMorphology(size=t_p, ellipticity=eps_p)()
    psf /= psf.sum()
    p = Moments(psf, N=2)
    psf = ArrayPSF(psf[None, ...])
    model_frame_ = scarlet2.Frame(scarlet2.Box(model.shape), psf=psf)

    # obs PSF
    t_obs, eps_obs = 3, jnp.array((0.1, -0.1))
    psf_obs = GaussianMorphology(size=t_obs, ellipticity=eps_obs)()
    psf_obs /= psf_obs.sum()
    p_obs = Moments(psf_obs, N=2)
    psf_obs = ArrayPSF(psf_obs[None, ...])

    obs_frame = scarlet2.Frame(scarlet2.Box(shape), psf=psf_obs, wcs=wcs_obs)
    renderer = cls(model_frame_, obs_frame)
    model_ = renderer(model)

    g_obs = Moments(model_[0])
    g_obs.deconvolve(p_obs).flipud().rotate(phi).resize(scale).convolve(p)
    shift_ = jnp.asarray(g_obs.centroid) - jnp.asarray(obs_frame.bbox.spatial.center)

    assert_allclose(g_obs.flux, g.flux, rtol=2e-3)
    assert_allclose(shift_, -shift, atol=1e-4)
    assert_allclose(g_obs.size, g.size, rtol=3e-5)
    assert_allclose(g_obs.ellipticity, g.ellipticity, atol=3e-5)
