"""Channels for multi-band and multi-epoch frames

The channels of a :py:class:`~scarlet2.Frame` can identify spectral bands, epochs, or both. A
:py:class:`Channels` instance holds the channel identifiers together with a `format`, which states what each
identifier contains:

* `BAND` (default): every channel is a band identifier, e.g. `["g", "r", "i"]`;
* `EPOCH`: every channel is an epoch identifier, e.g. `[0, 1, 2]`;
* a tuple of channel types: every channel is a tuple with the components in that order, e.g.
  `(BAND, EPOCH)` for `[("g", 0), ("g", 1), ("r", 0)]`.

Examples
--------
>>> channels = Channels([("g", 0), ("g", 1), ("r", 0)], (BAND, EPOCH))
>>> channels.get_component(BAND)
['g', 'g', 'r']
>>> [c.epoch for c in channels]
[0, 1, 0]
>>> channels[2]
Channel(band='r', epoch=0)
>>> print(channels[2])
('r', 0)
"""

import dataclasses
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


@dataclasses.dataclass(frozen=True)
class Channel:
    """Single channel of :py:class:`Channels`

    Created on the fly by :py:meth:`Channels.__getitem__`. Two channels are equal if :py:attr:`band` and
    :py:attr:`epoch` are equal, irrespective of the channel format.
    """

    band: object = None
    """Spectral band, `None` if the channel format does not contain `BAND`"""
    epoch: object = None
    """Epoch, `None` if the channel format does not contain `EPOCH`"""
    format: object = dataclasses.field(default=BAND, compare=False, repr=False)
    """Channel format of the identifier"""

    @property
    def value(self):
        """Channel identifier in the notation of :py:attr:`format`"""
        parts = {BAND: self.band, EPOCH: self.epoch}
        if isinstance(self.format, tuple):
            return tuple(parts[t] for t in self.format)
        return parts[self.format]

    def __str__(self):
        return str(self.value)


class Channels:
    """Channel identifiers of a :py:class:`~scarlet2.Frame`, together with their format

    Parameters
    ----------
    channels: list
        Channel identifiers. They do not need to be unique.
    channel_format: ChannelType, str, or tuple, optional
        Format of the identifiers, see :py:func:`check_channel_format`

    Raises
    ------
    ValueError
        If `channel_format` is invalid, or if `channels` do not match `channel_format`
    """

    def __init__(self, channels, channel_format=BAND):
        self.format = check_channel_format(channel_format)
        """Channel type, or tuple of channel types, contained in each identifier"""
        self.values = list(channels)
        """Channel identifiers"""
        if isinstance(self.format, tuple):
            for v in self.values:
                if not isinstance(v, tuple) or len(v) != len(self.format):
                    raise ValueError(f"Channel {v!r} does not match channel format {self.format!r}")

    @property
    def _types(self):
        return self.format if isinstance(self.format, tuple) else (self.format,)

    def get_component(self, kind):
        """Get the component of type `kind` for every channel

        Parameters
        ----------
        kind: ChannelType or str
            Type of the component, e.g. `BAND` or `EPOCH`

        Returns
        -------
        list or None
            Component of every channel, in the order of the channels, or `None` if :py:attr:`format` does not
            contain `kind`
        """
        kind = _to_type(kind)
        if kind not in self._types:
            return None
        if isinstance(self.format, tuple):
            i = self.format.index(kind)
            return [v[i] for v in self.values]
        return list(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Channels(self.values[i], self.format)
        v = self.values[i]
        parts = dict(zip(self._types, v if isinstance(self.format, tuple) else (v,), strict=True))
        return Channel(band=parts.get(BAND), epoch=parts.get(EPOCH), format=self.format)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def index(self, channel):
        """Index of the first occurrence of `channel`

        Raises
        ------
        ValueError
            If `channel` is not present
        """
        return list(self).index(channel)

    def __add__(self, other):
        if not isinstance(other, Channels):
            return NotImplemented
        if self.format != other.format:
            raise ValueError(f"Cannot combine channel formats {self.format!r} and {other.format!r}")
        return Channels(self.values + other.values, self.format)

    def __eq__(self, other):
        if not isinstance(other, Channels):
            try:
                other = Channels(other)
            except (TypeError, ValueError):
                return NotImplemented
        return list(self) == list(other)

    def __hash__(self):
        # consistent with __eq__, which ignores the format
        return hash(tuple(self))

    def __repr__(self):
        return f"Channels({self.values!r}, format={self.format!r})"


def unique(values):
    """Unique elements of `values`, in the order of their first appearance

    Examples
    --------
    >>> unique(["g", "g", "r", "g"])
    ['g', 'r']
    """
    return list(dict.fromkeys(values))
