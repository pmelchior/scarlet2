# ruff: noqa: D103

"""Tests for :py:func:`scarlet2.stack_observations`."""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from scarlet2 import (
    ArrayPSF,
    CorrelatedObservation,
    Frame,
    GaussianPSF,
    Observation,
    Scene,
    Source,
    stack_observations,
)
from scarlet2.bbox import Box
from scarlet2.frame import _wcs_default
from scarlet2.morphology import GaussianMorphology
from scarlet2.observation import _informative_mode_mask
from scarlet2.renderer import ChannelTransformation, ConvolutionTransformation
from scarlet2.validation_utils import set_validation

NF = 64  # model-frame size
OFFSET = (7, 11)  # integer pixel offset of the sub-grid observation
SUB = (40, 44)  # sub-grid observation size


@pytest.fixture(autouse=True)
def _no_validation():
    set_validation(False)


def _gauss_psf(sigma, nchan=1, shape=(15, 15)):
    yy, xx = np.indices(shape).astype(float)
    yy -= shape[0] // 2
    xx -= shape[1] // 2
    g = np.exp(-(yy**2 + xx**2) / (2 * sigma**2))
    return ArrayPSF(jnp.tile(jnp.asarray(g / g.sum())[None], (nchan, 1, 1)))


def _sub_wcs(model_wcs, origin, shape):
    """WCS for a sub-grid whose pixel (0, 0) sits at model pixel `origin`."""
    wcs = model_wcs.deepcopy()
    wcs._naxis = list(wcs._naxis)
    wcs._naxis[:2] = shape[::-1]  # x/y
    wcs.wcs.crpix[:2] -= np.array(origin[::-1])  # x/y
    return wcs


@pytest.fixture()
def setup():
    rng = np.random.default_rng(0)
    model_wcs = _wcs_default((NF, NF))
    model_frame = Frame(
        Box((5, NF, NF)), psf=GaussianPSF(0.7), wcs=model_wcs, channels=["u", "g", "r", "i", "z"]
    )

    obs_g = Observation(
        jnp.asarray(rng.standard_normal((1, NF, NF))),
        weights=jnp.full((1, NF, NF), 4.0),
        psf=_gauss_psf(1.2, 1),
        wcs=model_wcs,
        channels=["g"],
        name="g-cam",
    )
    sub_wcs = _sub_wcs(model_wcs, OFFSET, SUB)
    obs_ri = Observation(
        jnp.asarray(rng.standard_normal((2, *SUB))),
        weights=jnp.full((2, *SUB), 2.0),
        psf=_gauss_psf(1.5, 2, shape=(19, 19)),  # larger PSF -> common-shape padding in stack()
        wcs=sub_wcs,
        channels=["r", "i"],
        name="ri-cam",
    )
    return model_frame, obs_g, obs_ri


def _scene(model_frame):
    with Scene(model_frame) as scene:
        Source(
            jnp.array([NF / 2.0, NF / 2.0]),
            jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            GaussianMorphology(3.0, shape=(NF, NF)),
        )
    return scene


def test_shapes_and_channels(setup):
    model_frame, obs_g, obs_ri = setup
    stacked = stack_observations([obs_g, obs_ri], model_frame)

    assert stacked.data.shape == (3, NF, NF)
    assert stacked.weights.shape == (3, NF, NF)
    assert stacked.frame.channels == ["g", "r", "i"]
    assert stacked.frame.psf().shape == (3, 19, 19)
    assert stacked.name == "g-cam + ri-cam"


def test_data_placement(setup):
    model_frame, obs_g, obs_ri = setup
    stacked = stack_observations([obs_g, obs_ri], model_frame)

    # full-grid observation lands unchanged
    assert_allclose(stacked.data[0], obs_g.data[0])
    assert_allclose(stacked.weights[0], obs_g.weights[0])

    # sub-grid observation lands at the integer offset, zero weight elsewhere
    oy, ox = OFFSET
    sl = (slice(oy, oy + SUB[0]), slice(ox, ox + SUB[1]))
    assert_allclose(stacked.data[1:][(slice(None), *sl)], obs_ri.data)
    assert_allclose(stacked.weights[1:][(slice(None), *sl)], obs_ri.weights)

    mask = np.ones((NF, NF), dtype=bool)
    mask[sl] = False
    assert np.all(np.asarray(stacked.weights[1])[mask] == 0)
    assert np.all(np.asarray(stacked.data[1])[mask] == 0)


def test_render_matches_individual(setup):
    model_frame, obs_g, obs_ri = setup
    scene = _scene(model_frame)
    model = scene()

    obs_g.match(model_frame)
    obs_ri.match(model_frame)
    stacked = stack_observations([obs_g, obs_ri], model_frame)
    stacked.match(model_frame)

    assert isinstance(stacked.renderer[0], ChannelTransformation)
    assert isinstance(stacked.renderer[1], ConvolutionTransformation)

    render = stacked.render(model)
    assert render.shape == (3, NF, NF)

    assert_allclose(render[0], obs_g.render(model)[0], atol=1e-5)

    oy, ox = OFFSET
    sl = (slice(None), slice(oy, oy + SUB[0]), slice(ox, ox + SUB[1]))
    assert_allclose(render[1:][sl], obs_ri.render(model), atol=1e-5)


def test_log_likelihood_finite(setup):
    model_frame, obs_g, obs_ri = setup
    scene = _scene(model_frame)
    stacked = stack_observations([obs_g, obs_ri], model_frame)
    stacked.match(model_frame)
    assert jnp.isfinite(stacked.log_likelihood(scene()))


def test_resamples_mismatched_grid(setup):
    model_frame, obs_g, _ = setup
    rng = np.random.default_rng(2)
    shifted = _wcs_default((NF, NF))
    shifted.wcs.crpix[0] += 0.5
    obs_bad = Observation(
        jnp.asarray(rng.standard_normal((1, NF, NF))),
        weights=jnp.full((1, NF, NF), 4.0),
        psf=_gauss_psf(1.0),
        wcs=shifted,
        channels=["z"],
    )
    stacked = stack_observations([obs_g, obs_bad], model_frame)

    # obs_bad sits off the model grid and gets resampled, which promotes the whole stack to a
    # correlated noise model even though obs_g itself matched the grid already
    assert isinstance(stacked, CorrelatedObservation)
    assert stacked.frame.channels == ["g", "z"]
    assert stacked.data.shape == (2, NF, NF)

    stacked.match(model_frame)
    scene = _scene(model_frame)
    assert jnp.isfinite(stacked.log_likelihood(scene()))


def test_rejects_duplicate_channels(setup):
    model_frame, obs_g, _ = setup
    with pytest.raises(AssertionError, match="duplicate channels"):
        stack_observations([obs_g, obs_g], model_frame)


# --- CorrelatedObservation path ---------------------------------------------------------------

CF = 56  # model-frame size for the correlated tests
CSUB = (36, 36)
COFF = (8, 10)


def _corr_fn(nchan, var):
    """Small symmetric correlation function: sharp core, weak nearest/diagonal neighbors."""
    lags = {(0, 0): var, (0, 1): 0.2 * var, (1, 0): 0.2 * var, (1, 1): 0.05 * var}
    xi = {}
    for (dy, dx), v in lags.items():
        arr = jnp.full((nchan,), float(v))
        xi[(dy, dx)] = arr
        if (dy, dx) != (0, 0):
            xi[(-dy, -dx)] = arr
    return xi


@pytest.fixture()
def corr_setup():
    rng = np.random.default_rng(1)
    model_wcs = _wcs_default((CF, CF))
    model_frame = Frame(Box((2, CF, CF)), psf=GaussianPSF(0.7), wcs=model_wcs, channels=["g", "r"])

    obs_g = CorrelatedObservation(
        jnp.asarray(rng.standard_normal((1, CF, CF))),
        psf=_gauss_psf(1.2, 1),
        wcs=model_wcs,
        channels=["g"],
        correlation_function=_corr_fn(1, 1.5),
        name="g-cam",
    )
    sub_wcs = _sub_wcs(model_wcs, COFF, CSUB)
    obs_r = CorrelatedObservation(
        jnp.asarray(rng.standard_normal((1, *CSUB))),
        psf=_gauss_psf(1.5, 1, shape=(17, 17)),
        wcs=sub_wcs,
        channels=["r"],
        correlation_function=_corr_fn(1, 2.5),
        name="r-cam",
    )
    return model_frame, obs_g, obs_r


def test_correlated_stack_structure(corr_setup):
    model_frame, obs_g, obs_r = corr_setup
    stacked = stack_observations([obs_g, obs_r], model_frame)

    assert isinstance(stacked, CorrelatedObservation)
    assert stacked.frame.channels == ["g", "r"]
    assert stacked.data.shape == (2, CF, CF)

    # n_eff carried through per channel: full grid vs sub grid
    assert stacked.n_eff == (obs_g.n_eff[0], obs_r.n_eff[0])
    assert stacked.n_eff == (CF * CF, CSUB[0] * CSUB[1])
    assert sum(stacked.n_eff) == stacked.N

    # correlation function concatenated over channels
    assert stacked.correlation_function[0, 0].shape == (2,)
    assert_allclose(stacked.correlation_function[0, 0], [1.5, 2.5])

    # padded region of the sub-grid channel is masked, the full channel is not
    assert not np.asarray(stacked.mask[0]).any()
    oy, ox = COFF
    inside = np.zeros((CF, CF), dtype=bool)
    inside[oy : oy + CSUB[0], ox : ox + CSUB[1]] = True
    assert np.all(~np.asarray(stacked.mask[1])[inside])
    assert np.all(np.asarray(stacked.mask[1])[~inside])


def test_correlated_stack_matches_and_scores(corr_setup):
    model_frame, obs_g, obs_r = corr_setup
    stacked = stack_observations([obs_g, obs_r], model_frame)
    stacked.match(model_frame)

    assert isinstance(stacked.renderer[-1], ConvolutionTransformation)

    with Scene(model_frame) as scene:
        Source(
            jnp.array([CF / 2.0, CF / 2.0]),
            jnp.array([1.0, 2.0]),
            GaussianMorphology(3.0, shape=(CF, CF)),
        )
    assert jnp.isfinite(stacked.log_likelihood(scene()))
    assert jnp.isfinite(stacked.goodness_of_fit(scene()))


def test_correlated_stack_missing_lags(corr_setup):
    """Observations with different lag ranges: the shorter one is zero-filled at the long lags."""
    model_frame, obs_g, obs_r = corr_setup
    long_xi = _corr_fn(1, 2.5)
    long_xi[(2, 0)] = long_xi[(-2, 0)] = jnp.full((1,), 0.1)
    obs_r_long = CorrelatedObservation(
        obs_r.data,
        psf=obs_r.frame.psf(),
        wcs=obs_r.frame.wcs,
        channels=["r"],
        correlation_function=long_xi,
        name="r-cam",
    )
    stacked = stack_observations([obs_g, obs_r_long], model_frame)
    assert (2, 0) in stacked.correlation_function
    assert_allclose(stacked.correlation_function[2, 0], [0.0, 0.1])


def test_promotes_mixed_types(setup):
    model_frame, obs_g, _ = setup
    corr_z = CorrelatedObservation(
        jnp.asarray(np.random.default_rng(3).standard_normal((1, NF, NF))),
        psf=_gauss_psf(1.0),
        wcs=model_frame.wcs,
        channels=["z"],
        correlation_function=_corr_fn(1, 1.0),
        name="z-cam",
    )
    stacked = stack_observations([obs_g, corr_z], model_frame)

    # obs_g already matched the grid and carries independent noise, but corr_z is correlated, so
    # obs_g is promoted to a CorrelatedObservation too
    assert isinstance(stacked, CorrelatedObservation)
    assert stacked.frame.channels == ["g", "z"]
    assert stacked.data.shape == (2, NF, NF)

    stacked.match(model_frame)
    scene = _scene(model_frame)
    assert jnp.isfinite(stacked.log_likelihood(scene()))


def test_informative_mode_mask_per_channel():
    ps = np.stack([np.arange(12).reshape(3, 4), np.arange(12)[::-1].reshape(3, 4)])
    mask = np.asarray(_informative_mode_mask(ps, [3, 5]))
    assert mask[0].sum() == 3
    assert mask[1].sum() == 5
    assert set(ps[0][mask[0]]) == {9, 10, 11}
    assert set(ps[1][mask[1]]) == {7, 8, 9, 10, 11}

    # a scalar still broadcasts across channels
    mask = np.asarray(_informative_mode_mask(ps, 4))
    assert mask[0].sum() == 4
    assert mask[1].sum() == 4
