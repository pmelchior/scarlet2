"""Channel types for multi-band and multi-epoch frames

The channels of a :py:class:`~scarlet2.Frame` can identify spectral bands, epochs, or both. The
`channel_format` of the frame states what each channel identifier contains:

* `BAND` (default): every channel is a band identifier, e.g. `channels=["g", "r", "i"]`;
* `EPOCH`: every channel is an epoch identifier, e.g. `channels=[0, 1, 2]`;
* a tuple of channel types: every channel is a tuple with the components in that order, e.g.
  `channel_format=(BAND, EPOCH)` for `channels=[("g", 0), ("g", 1), ("r", 0)]`.
"""

from enum import Enum


class ChannelType(str, Enum):
    """Type of information stored in a channel identifier"""

    BAND = "band"
    """Spectral band"""
    EPOCH = "epoch"
    """Epoch of the observation"""

    def __repr__(self):
        return self.name


BAND = ChannelType.BAND
EPOCH = ChannelType.EPOCH


def check_channel_format(channel_format):
    """Check and normalize a channel format

    Parameters
    ----------
    channel_format: ChannelType, str, or tuple
        Single channel type, or tuple of distinct channel types. The strings `"band"` and `"epoch"`
        are accepted in place of `BAND` and `EPOCH`.

    Returns
    -------
    ChannelType or tuple of ChannelType

    Raises
    ------
    ValueError
        If `channel_format` is not a valid channel format

    Examples
    --------
    >>> check_channel_format("band")
    BAND
    >>> check_channel_format((BAND, "epoch"))
    (BAND, EPOCH)
    """
    if isinstance(channel_format, tuple):
        if len(channel_format) == 0:
            raise ValueError("Channel format cannot be an empty tuple")
        normalized = tuple(_to_type(t) for t in channel_format)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Channel format {channel_format!r} contains repeated channel types")
        return normalized
    return _to_type(channel_format)


def _to_type(t):
    try:
        return ChannelType(t)
    except ValueError:
        raise ValueError(
            f"Unknown channel type {t!r}, allowed are {[t.value for t in ChannelType]}"
        ) from None


def check_channels(channels, channel_format):
    """Check that `channels` agree with `channel_format`

    For a tuple format, every channel needs to be a tuple of the same length. Channels do not need to be
    unique.

    Raises
    ------
    ValueError
        If a channel does not match `channel_format`
    """
    if isinstance(channel_format, tuple):
        for c in channels:
            if not isinstance(c, tuple) or len(c) != len(channel_format):
                raise ValueError(f"Channel {c!r} does not match channel format {channel_format!r}")


def get_component(channel, channel_format, kind):
    """Get the component of type `kind` from `channel`

    Returns
    -------
    Component of `channel`, or `None` if `channel_format` does not contain `kind`

    Examples
    --------
    >>> get_component(("g", 1), (BAND, EPOCH), EPOCH)
    1
    >>> get_component("g", BAND, EPOCH) is None
    True
    """
    kind = _to_type(kind)
    if isinstance(channel_format, tuple):
        if kind not in channel_format:
            return None
        return channel[channel_format.index(kind)]
    return channel if channel_format == kind else None


def unique(values):
    """Unique elements of `values`, in the order of their first appearance

    Examples
    --------
    >>> unique(["g", "g", "r", "g"])
    ['g', 'r']
    """
    return list(dict.fromkeys(values))
