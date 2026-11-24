import pickle

import equinox as eqx
import jax.numpy as jnp
import pytest

from scarlet2 import (
    BAND,
    EPOCH,
    Channel,
    Channels,
    Frame,
    Observation,
    Scene,
    StaticArraySpectrum,
    TransientArraySpectrum,
)
from scarlet2.bbox import Box
from scarlet2.channel import check_channel_format


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
    with pytest.raises(ValueError):
        Channels(["g"], channel_format)


@pytest.mark.parametrize(
    "values",
    [["g", "r"], [("g", 0), ("g", 1, 2)], [["g", 0]]],
)
def test_channel_mismatch(values):
    """Test that channels not matching the format are rejected"""
    with pytest.raises(ValueError):
        Channels(values, (BAND, EPOCH))


def test_get_component():
    """Test per-channel components"""
    values = [("g", 0), ("g", 1), ("r", 0), ("i", 2)]
    channels = Channels(values, (BAND, EPOCH))
    assert channels.values == values
    assert channels.get_component(BAND) == ["g", "g", "r", "i"]
    assert channels.get_component(EPOCH) == [0, 1, 0, 2]
    assert channels.get_component("epoch") == [0, 1, 0, 2]

    channels = Channels([(0, "g"), (1, "r")], (EPOCH, BAND))
    assert channels.get_component(BAND) == ["g", "r"]
    assert channels.get_component(EPOCH) == [0, 1]

    # default: channels are bands, no epochs
    channels = Channels(["i", "g", "r"])
    assert channels.format == BAND
    assert channels.get_component(BAND) == ["i", "g", "r"]
    assert channels.get_component(EPOCH) is None

    # epochs only
    channels = Channels([0, 1, 2], EPOCH)
    assert channels.get_component(BAND) is None
    assert channels.get_component(EPOCH) == [0, 1, 2]

    # channels do not need to be unique
    channels = Channels(["g", "g", "r"])
    assert channels.get_component(BAND) == ["g", "g", "r"]


def test_channel_access():
    """Test indexing and iteration, which create Channel instances"""
    channels = Channels([("g", 0), ("g", 1), ("r", 0)], (BAND, EPOCH))
    assert len(channels) == 3
    assert channels[2] == Channel(band="r", epoch=0)
    assert channels[-1] == Channel(band="r", epoch=0)
    assert [c.epoch for c in channels] == [0, 1, 0]
    assert channels[1:] == Channels([("g", 1), ("r", 0)], (BAND, EPOCH))
    assert channels.index(Channel(band="g", epoch=1)) == 1
    assert Channel(band="r", epoch=0) in channels

    # repr shows the components, str the identifier in the user's notation
    assert repr(channels[0]) == "Channel(band='g', epoch=0)"
    assert str(channels[0]) == "('g', 0)"
    assert str(Channels([(0, "g")], (EPOCH, BAND))[0]) == "(0, 'g')"
    assert str(Channels(["g"])[0]) == "g"
    assert Channels(["g"])[0] == Channel(band="g")
    assert Channels([3], EPOCH)[0] == Channel(epoch=3)


def test_channels_equality():
    """Test equality, hashing, and combination of Channels"""
    c1 = Channels([("g", 0), ("r", 1)], (BAND, EPOCH))
    c2 = Channels([(0, "g"), (1, "r")], (EPOCH, BAND))
    # equality and hash ignore the format
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 != Channels([("g", 0), ("r", 2)], (BAND, EPOCH))

    # comparison with a list assumes the BAND format
    assert Channels(["g", "r"]) == ["g", "r"]
    assert Channels(["g", "r"]) != ["r", "g"]
    assert c1 != ["g", "r"]

    combined = c1 + Channels([("i", 2)], (BAND, EPOCH))
    assert combined.values == [("g", 0), ("r", 1), ("i", 2)]
    with pytest.raises(ValueError):
        c1 + c2

    assert pickle.loads(pickle.dumps(c1)) == c1


def test_frame_channels():
    """Test channels of Frame"""
    channels = Channels([("g", 0), ("g", 1), ("r", 0)], (BAND, EPOCH))
    frame = Frame(Box((3, 10, 10)), channels=channels)
    assert frame.channels is channels
    assert frame.C == 3

    # flat list: BAND format
    frame = Frame(Box((3, 10, 10)), channels=["i", "g", "r"])
    assert isinstance(frame.channels, Channels)
    assert frame.channels.format == BAND
    assert frame.channels == ["i", "g", "r"]
    frame = Frame(Box((3, 10, 10)))
    assert frame.channels.get_component(BAND) == [0, 1, 2]

    # channels are static, i.e. not part of the pytree leaves
    leaves = eqx.filter(frame, eqx.is_array)
    assert leaves.channels is frame.channels

    frame_ = pickle.loads(pickle.dumps(frame))
    assert frame_.channels == frame.channels


def test_from_observations_channels():
    """Test combination of channels from several observations"""
    data = jnp.zeros((2, 11, 11))
    psf = jnp.ones((2, 5, 5)) / 25
    obs1 = Observation(
        data, jnp.ones_like(data), psf=psf, channels=Channels([("g", 0), ("r", 0)], (BAND, EPOCH))
    )
    obs2 = Observation(
        data, jnp.ones_like(data), psf=psf, channels=Channels([("g", 1), ("r", 1)], (BAND, EPOCH))
    )
    model_frame = Frame.from_observations([obs1, obs2])
    assert model_frame.channels.format == (BAND, EPOCH)
    assert model_frame.channels.values == [("g", 0), ("r", 0), ("g", 1), ("r", 1)]
    assert model_frame.channels.get_component(EPOCH) == [0, 0, 1, 1]

    # different channel formats cannot be combined
    obs3 = Observation(
        data, jnp.ones_like(data), psf=psf, channels=Channels([(2, "g"), (2, "r")], (EPOCH, BAND))
    )
    with pytest.raises(ValueError):
        Frame.from_observations([obs1, obs3])


def test_spectra():
    """Test static and transient spectra with channel formats"""
    channels = Channels([("G", 0), ("G", 1), ("R", 0), ("R", 1), ("R", 2)], (BAND, EPOCH))
    frame = Frame(Box((5, 10, 10)), channels=channels)
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
