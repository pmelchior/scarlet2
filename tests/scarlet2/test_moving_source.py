import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from scarlet2 import (
    BAND,
    EPOCH,
    Channels,
    Component,
    Frame,
    GaussianMorphology,
    GaussianPSF,
    MovingPointSource,
    MovingSource,
    Observation,
    Parameter,
    Parameters,
    PointSource,
    Scene,
    Source,
)
from scarlet2.bbox import Box
from scarlet2.validation_utils import set_validation

CHANNELS = [("g", 0), ("r", 0), ("g", 1), ("r", 1), ("g", 2)]
CENTERS = jnp.array([[12.0, 10.0], [15.3, 16.0], [18.0, 21.7]])
SPECTRUM = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
EPOCHS = [0, 1, 2]


@pytest.fixture
def frame():
    """Model frame with three epochs in two bands"""
    return Frame(Box((5, 32, 32)), channels=Channels(CHANNELS, (BAND, EPOCH)))


def _epoch_of_channel(c):
    return CHANNELS[c][1]


def test_moving_source_slices(frame):
    """Every channel holds the morphology at the center of its epoch, scaled by the spectrum"""
    morph = GaussianMorphology(1.0)
    with Scene(frame) as scene:
        src = MovingSource(CENTERS, SPECTRUM, morph, epochs=EPOCHS)

    assert src.epochs == (0, 1, 2)
    assert src.bbox.shape[0] == len(CHANNELS)
    model = scene()
    assert model.shape == frame.bbox.shape
    for c in range(len(CHANNELS)):
        center = CENTERS[_epoch_of_channel(c)]
        peak = np.unravel_index(np.argmax(model[c]), model[c].shape)
        assert_allclose(peak, np.round(center), atol=0)
        delta = center - center.astype(int)
        assert_allclose(model[c].sum(), SPECTRUM[c] * morph(delta_center=delta).sum(), rtol=1e-5)


def test_moving_source_matches_components(frame):
    """Moving source equals the naive sum of one-hot components, one per channel"""
    morph = GaussianMorphology(1.0)
    with Scene(frame) as scene:
        MovingSource(CENTERS, SPECTRUM, morph, epochs=EPOCHS)
    model = scene()

    with Scene(frame) as ref_scene:
        for c in range(len(CHANNELS)):
            spectrum = jnp.zeros(len(CHANNELS)).at[c].set(SPECTRUM[c])
            Source(CENTERS[_epoch_of_channel(c)], spectrum, morph)
    assert_allclose(model, ref_scene(), atol=1e-6)


def test_moving_source_per_channel(frame):
    """Without epochs, every channel has its own center"""
    centers = jnp.array([[8.0, 8.0], [12.0, 10.0], [15.3, 16.0], [18.0, 21.7], [22.0, 24.0]])
    with Scene(frame) as scene:
        src = MovingSource(centers, SPECTRUM, GaussianMorphology(1.0))
    assert src.epochs is None
    model = scene()
    for c in range(len(CHANNELS)):
        peak = np.unravel_index(np.argmax(model[c]), model[c].shape)
        assert_allclose(peak, np.round(centers[c]))

    # same as explicit epochs if every channel is its own epoch
    with Scene(frame) as ref_scene:
        MovingSource(CENTERS[np.array([0, 0, 1, 1, 2])], SPECTRUM, GaussianMorphology(1.0))
    with Scene(frame) as epoch_scene:
        MovingSource(CENTERS, SPECTRUM, GaussianMorphology(1.0), epochs=EPOCHS)
    assert_allclose(ref_scene(), epoch_scene(), atol=1e-6)


def test_moving_source_band_only():
    """Without EPOCH in the channel format, centers are per channel"""
    frame = Frame(Box((3, 32, 32)), channels=Channels(["g", "r", "i"]))
    with Scene(frame) as scene:
        MovingSource(CENTERS, SPECTRUM[:3], GaussianMorphology(1.0))
    model = scene()
    for c in range(3):
        peak = np.unravel_index(np.argmax(model[c]), model[c].shape)
        assert_allclose(peak, np.round(CENTERS[c]))

    # epochs cannot be mapped without EPOCH
    with Scene(frame), pytest.raises(ValueError, match="no EPOCH"):
        MovingSource(CENTERS, SPECTRUM[:3], GaussianMorphology(1.0), epochs=EPOCHS)


def test_moving_source_explicit_epochs(frame):
    """Explicit epoch order permutes the centers"""
    with Scene(frame) as scene:
        MovingSource(CENTERS[::-1], SPECTRUM, GaussianMorphology(1.0), epochs=[2, 1, 0])
    with Scene(frame) as ref_scene:
        MovingSource(CENTERS, SPECTRUM, GaussianMorphology(1.0), epochs=EPOCHS)
    assert_allclose(scene(), ref_scene(), atol=1e-6)


def test_moving_source_wrong_centers(frame):
    """Centers or spectrum with the wrong length raise errors"""
    with Scene(frame), pytest.raises(ValueError):
        MovingSource(CENTERS[:2], SPECTRUM, GaussianMorphology(1.0), epochs=EPOCHS)
    with Scene(frame), pytest.raises(ValueError):
        MovingSource(CENTERS, SPECTRUM[:3], GaussianMorphology(1.0), epochs=EPOCHS)
    # without epochs, one center per channel is needed
    with Scene(frame), pytest.raises(ValueError):
        MovingSource(CENTERS, SPECTRUM, GaussianMorphology(1.0))
    # every epoch of the frame channels needs to be listed
    with Scene(frame), pytest.raises(ValueError, match="not in"):
        MovingSource(CENTERS[:2], SPECTRUM, GaussianMorphology(1.0), epochs=[0, 1])


def test_moving_source_skycoord():
    """Centers given as array of sky coordinates are converted to pixels"""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [16.0, 16.0]
    wcs.wcs.cdelt = [-0.2 / 3600, 0.2 / 3600]
    frame = Frame(Box((5, 32, 32)), wcs=wcs, channels=Channels(CHANNELS, (BAND, EPOCH)))
    sky = frame.get_sky_coord(CENTERS)
    with Scene(frame):
        src = MovingSource(sky, SPECTRUM, GaussianMorphology(1.0), epochs=EPOCHS)
    assert_allclose(src.center, CENTERS, atol=1e-4)


def test_moving_point_source():
    """Moving point source equals one point source per channel"""
    frame = Frame(Box((5, 32, 32)), psf=GaussianPSF(0.8), channels=Channels(CHANNELS, (BAND, EPOCH)))
    with Scene(frame) as scene:
        MovingPointSource(CENTERS, SPECTRUM, epochs=EPOCHS)
    with Scene(frame) as ref_scene:
        for c in range(len(CHANNELS)):
            spectrum = jnp.zeros(len(CHANNELS)).at[c].set(SPECTRUM[c])
            PointSource(CENTERS[_epoch_of_channel(c)], spectrum)
    assert_allclose(scene(), ref_scene(), atol=1e-6)


def test_moving_source_with_component(frame):
    """Additional components combine with the moving base model"""
    with Scene(frame) as scene:
        src = MovingSource(CENTERS, SPECTRUM, GaussianMorphology(1.0), epochs=EPOCHS)
        host = Component(jnp.array([16.0, 16.0]), jnp.ones(5), GaussianMorphology(2.0))
        src += host
    with Scene(frame) as ref_scene:
        MovingSource(CENTERS, SPECTRUM, GaussianMorphology(1.0), epochs=EPOCHS)
        Source(jnp.array([16.0, 16.0]), jnp.ones(5), GaussianMorphology(2.0))
    assert len(scene.sources) == 1
    assert_allclose(scene(), ref_scene(), atol=1e-6)


@pytest.fixture
def validation_on():
    """Enable validation for one test, restoring the previous mode afterwards"""
    from scarlet2 import validation_utils

    previous = validation_utils.VALIDATION_MODE
    set_validation(True)
    yield
    set_validation(previous)


def test_moving_source_fit(frame, validation_on):
    """Gradients reach the centers, and a fit recovers perturbed centers"""
    # source box with border extends beyond the frame: tests the chi^2 validation in fit()
    centers = CENTERS
    with Scene(frame) as truth:
        MovingSource(centers, SPECTRUM, GaussianMorphology(1.0), epochs=EPOCHS)
    data = truth()
    obs = Observation(data, weights=jnp.ones_like(data) * 100, channels=frame.channels)

    perturbed = centers + jnp.array([[0.4, -0.3], [-0.3, 0.3], [0.3, 0.2]])
    with Scene(frame) as scene:
        MovingSource(perturbed, SPECTRUM, GaussianMorphology(1.0), epochs=EPOCHS)

    grad = jax.grad(lambda c: ((scene.replace_subtree(lambda s: s.sources[0].center, c)() - data) ** 2).sum())
    g = grad(scene.sources[0].center)
    assert jnp.all(jnp.isfinite(g)) and jnp.all(jnp.abs(g) > 0)

    with Parameters(scene):
        Parameter(scene.sources[0].center, name="center", stepsize=0.05)
    scene_ = scene.fit(obs, max_iter=300, e_rel=1e-6, progress_bar=False)
    assert_allclose(scene_.sources[0].center, centers, atol=0.05)
