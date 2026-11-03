import pickle

import jax.numpy as jnp
import pytest

from scarlet2 import BAND, EPOCH, Frame, Observation, Scene, StaticArraySpectrum, TransientArraySpectrum
from scarlet2.bbox import Box
from scarlet2.channel import check_channel_format, check_channels


@pytest.mark.parametrize(
    "channel_format, expected",
    [
        (BAND, BAND),
        ("band", BAND),
        (EPOCH, EPOCH),
        ((BAND, EPOCH), (BAND, EPOCH)),
        (("epoch", "band"), (EPOCH, BAND)),
        ((BAND,), (BAND,)),
    ],
)
def test_valid_channel_format(channel_format, expected):
    """Test normalization of valid channel formats"""
    assert check_channel_format(channel_format) == expected


@pytest.mark.parametrize(
    "channel_format",
    ["", "filter", (), (BAND, BAND), (BAND, "time"), [BAND, EPOCH], None],
)
def test_invalid_channel_format(channel_format):
    """Test that invalid channel formats are rejected"""
    with pytest.raises(ValueError):
        check_channel_format(channel_format)


@pytest.mark.parametrize(
    "channels, channel_format",
    [
        (["g", "r"], (BAND, EPOCH)),
        ([("g", 0), ("g", 1, 2)], (BAND, EPOCH)),
        ([["g", 0]], (BAND, EPOCH)),
    ],
)
def test_channel_mismatch(channels, channel_format):
    """Test that channels not matching the format are rejected"""
    with pytest.raises(ValueError):
        check_channels(channels, channel_format)
    with pytest.raises(ValueError):
        Frame(Box((len(channels), 10, 10)), channels=channels, channel_format=channel_format)


def test_get_component():
    """Test per-channel components of Frame"""
    channels = [("g", 0), ("g", 1), ("r", 0), ("i", 2)]
    frame = Frame(Box((4, 10, 10)), channels=channels, channel_format=(BAND, EPOCH))
    assert frame.channels == channels
    assert frame.get_component(BAND) == ["g", "g", "r", "i"]
    assert frame.get_component(EPOCH) == [0, 1, 0, 2]
    assert frame.get_component("epoch") == [0, 1, 0, 2]

    frame = Frame(Box((2, 10, 10)), channels=[(0, "g"), (1, "r")], channel_format=(EPOCH, BAND))
    assert frame.get_component(BAND) == ["g", "r"]
    assert frame.get_component(EPOCH) == [0, 1]

    # default: channels are bands, no epochs
    frame = Frame(Box((3, 10, 10)), channels=["i", "g", "r"])
    assert frame.channel_format == BAND
    assert frame.get_component(BAND) == ["i", "g", "r"]
    assert frame.get_component(EPOCH) is None
    frame = Frame(Box((3, 10, 10)))
    assert frame.get_component(BAND) == [0, 1, 2]

    # epochs only
    frame = Frame(Box((3, 10, 10)), channels=[0, 1, 2], channel_format=EPOCH)
    assert frame.get_component(BAND) is None
    assert frame.get_component(EPOCH) == [0, 1, 2]

    # channels do not need to be unique
    frame = Frame(Box((3, 10, 10)), channels=["g", "g", "r"])
    assert frame.get_component(BAND) == ["g", "g", "r"]


def test_pickle():
    """Test pickling of frames with channel formats"""
    frame = Frame(Box((2, 10, 10)), channels=[("g", 0), ("r", 1)], channel_format=(BAND, EPOCH))
    frame_ = pickle.loads(pickle.dumps(frame))
    assert frame_.channels == frame.channels
    assert frame_.channel_format == frame.channel_format


def test_from_observations_channels():
    """Test combination of channels from several observations"""
    data = jnp.zeros((2, 11, 11))
    psf = jnp.ones((2, 5, 5)) / 25
    obs1 = Observation(
        data, jnp.ones_like(data), psf=psf, channels=[("g", 0), ("r", 0)], channel_format=(BAND, EPOCH)
    )
    obs2 = Observation(
        data, jnp.ones_like(data), psf=psf, channels=[("g", 1), ("r", 1)], channel_format=(BAND, EPOCH)
    )
    model_frame = Frame.from_observations([obs1, obs2])
    assert model_frame.channel_format == (BAND, EPOCH)
    assert model_frame.channels == [("g", 0), ("r", 0), ("g", 1), ("r", 1)]
    assert model_frame.get_component(BAND) == ["g", "r", "g", "r"]
    assert model_frame.get_component(EPOCH) == [0, 0, 1, 1]

    # different channel formats cannot be combined
    obs3 = Observation(
        data, jnp.ones_like(data), psf=psf, channels=[(2, "g"), (2, "r")], channel_format=(EPOCH, BAND)
    )
    with pytest.raises(ValueError):
        Frame.from_observations([obs1, obs3])


def test_spectra():
    """Test static and transient spectra with channel formats"""
    channels = [("G", 0), ("G", 1), ("R", 0), ("R", 1), ("R", 2)]
    frame = Frame(Box((5, 10, 10)), channels=channels, channel_format=(BAND, EPOCH))
    with Scene(frame):
        static = StaticArraySpectrum(jnp.array([1.0, 2.0]))
        transient = TransientArraySpectrum(jnp.arange(1.0, 5.0), epochs=[0, 1])
        always_on = TransientArraySpectrum(jnp.arange(1.0, 6.0))
    assert static.bands == ["G", "R"]
    assert jnp.all(static() == jnp.array([1, 1, 2, 2, 2]))
    assert jnp.all(transient() == jnp.array([1, 2, 3, 4, 0]))
    assert jnp.all(always_on() == jnp.arange(1.0, 6.0))

    # frame without epochs
    frame = Frame(Box((2, 10, 10)), channels=["G", "R"])
    with Scene(frame), pytest.raises(ValueError):
        TransientArraySpectrum(jnp.ones(2), epochs=[0])

    # deprecated selectors operate on the channel identifiers
    frame = Frame(Box((3, 10, 10)), channels=[("G", 0), ("G", 1), ("R", 0)])
    with Scene(frame), pytest.warns(DeprecationWarning):
        static = StaticArraySpectrum(jnp.array([1.0, 2.0]), ["G", "R"], band_selector=lambda c: c[0])
    assert jnp.all(static() == jnp.array([1, 1, 2]))
