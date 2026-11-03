from warnings import warn

import equinox as eqx
import jax.numpy as jnp

from . import Scenery
from .channel import BAND, EPOCH, unique
from .module import Module


class Spectrum(Module):
    """Spectrum base class"""

    @property
    def shape(self):
        """Shape (1D) of the spectrum model"""
        raise NotImplementedError


class StaticArraySpectrum(Spectrum):
    """Static (non-variable) source in a transient scene

    In the frames of transient scenes, the attribute :py:attr:`~scarlet2.Frame.channels`
    has a spectral and a temporal component, e.g. `channel_format=(BAND, EPOCH)`.
    This class is for models that do not vary in time, i.e. only have a spectral dependency.
    The length of :py:attr:`data` is thus given by the number of distinct spectral bands.
    """

    data: jnp.array
    """Data to describe the static spectrum

    The order in this array should be given by :py:attr:`bands`.
    """
    bands: list
    """Identifier for the list of unique bands in the model frame channels"""
    _channelindex: jnp.array = eqx.field(repr=False)

    def __init__(self, data, bands=None, band_selector=None):
        """
        Parameters
        ----------
        data: array
            Spectrum without temporal variation. Contains as many elements as there
            are spectral channels in the model.
        bands: list, array, optional
            Identifier for the list of unique bands in the model frame channels.
            If `None`, uses the unique bands of the model frame channels, in their order of appearance.
        band_selector: callable, optional
            Deprecated: bands are taken from the `BAND` component of the model frame channels.
            If set, identifies the spectral "band" component from a channel identifier.

        Examples
        --------
        >>> from scarlet2 import BAND, EPOCH, Frame, Scene
        >>> from scarlet2.bbox import Box
        >>> channels = [('G',0),('G',1),('R',0),('R',1),('R',2)]
        >>> frame = Frame(Box((5, 10, 10)), channels=channels, channel_format=(BAND, EPOCH))
        >>> with Scene(frame):
        ...     spectrum = StaticArraySpectrum(jnp.array([1., 2.]))
        >>> spectrum.bands
        ['G', 'R']
        >>> spectrum()
        Array([1., 1., 2., 2., 2.], dtype=float32)

        This constructs a 2-element spectrum to describe the spectral properties
        in all epochs 0,1,2.

        See Also
        --------
        TransientArraySpectrum
        """
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

        if band_selector is None:
            channel_bands = frame.get_component(BAND)
            if channel_bands is None:
                raise ValueError(f"Channel format {frame.channel_format!r} of the model frame has no BAND")
        else:
            warn(
                "`band_selector` is deprecated, use `channel_format` of the frame to define the bands",
                DeprecationWarning,
                stacklevel=2,
            )
            channel_bands = [band_selector(c) for c in frame.channels]
        bands = unique(channel_bands) if bands is None else list(bands)

        assert len(data) == len(bands)
        self.data = data
        self.bands = bands
        self._channelindex = jnp.array([self.bands.index(band) for band in channel_bands])

    def __call__(self):
        """What to run when the StaticArraySpectrum is called"""
        return self.data[self._channelindex]

    @property
    def shape(self):
        """The shape of the spectrum data"""
        return (len(self._channelindex),)


class TransientArraySpectrum(Spectrum):
    """Variable source in a transient scene with possible quiescent periods

    In the frames of transient scenes, the attribute :py:attr:`~scarlet2.Frame.channels`
    has a spectral and a temporal component, e.g. `channel_format=(BAND, EPOCH)`. This class is for models that vary in time, especially
    if they have periods of inactivity. The length of :py:attr:`data` is given by the
    number of active channels in the model frame; calling this spectrum reinserts a
    zero for every inactive (quiescent) channel, so that the result matches the full
    length and order of the model frame's channels.
    """

    data: jnp.array
    """Data to describe the variable spectrum, restricted to the active channels.

    The order in this array follows the order of the active (i.e. non-quiescent)
    channels in the model frame, i.e. those for which :py:attr:`epochs` selects an
    active epoch.
    """
    epochs: list
    """Identifier for the list of active epochs. If set to `None`, all epochs are
    considered active"""
    _channelindex: jnp.array = eqx.field(repr=False)

    def __init__(self, data, epochs=None, epoch_selector=None):
        """
        Parameters
        ----------
        data: array
            Spectrum array, restricted to the active epochs. Contains as many elements
            as there are spectro-temporal channels in the model whose epoch is in `epochs`.
        epochs: list, array, optional
            List of temporal "epoch" identifiers for the active phases of the source.
            If `None`, all epochs are considered active.
        epoch_selector: callable, optional
            Deprecated: epochs are taken from the `EPOCH` component of the model frame channels.
            If set, identifies the temporal "epoch" component from a channel identifier.

        Examples
        --------
        >>> from scarlet2 import BAND, EPOCH, Frame, Scene
        >>> from scarlet2.bbox import Box
        >>> channels = [('G',0),('G',1),('R',0),('R',1),('R',2)]
        >>> frame = Frame(Box((5, 10, 10)), channels=channels, channel_format=(BAND, EPOCH))
        >>> with Scene(frame):
        ...     # one value per active (band, epoch) channel
        ...     spectrum = TransientArraySpectrum(jnp.ones(4), epochs=[0, 1])
        >>> spectrum()
        Array([1., 1., 1., 1., 0.], dtype=float32)

        This defines a spectrum with one free parameter for every channel active
        during epochs 0 and 1, i.e. all but `('R',2)`. Calling this spectrum
        reinserts a zero for that inactive channel, so that the returned vector
        again has 5 elements, matching the model frame.

        See Also
        --------
        StaticArraySpectrum
        """
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

        self.epochs = epochs
        if epochs is None:
            is_active = [True] * len(frame.channels)
        else:
            if epoch_selector is None:
                channel_epochs = frame.get_component(EPOCH)
                if channel_epochs is None:
                    raise ValueError(
                        f"Channel format {frame.channel_format!r} of the model frame has no EPOCH"
                    )
            else:
                warn(
                    "`epoch_selector` is deprecated, use `channel_format` of the frame to define the epochs",
                    DeprecationWarning,
                    stacklevel=2,
                )
                channel_epochs = [epoch_selector(c) for c in frame.channels]
            is_active = [epoch in epochs for epoch in channel_epochs]
        n_active = sum(is_active)
        assert len(data) == n_active
        self.data = data

        # index of every channel into `data`; inactive channels are pointed at the
        # trailing zero-padded slot appended to `data` in __call__
        indices = []
        next_index = 0
        for active in is_active:
            if active:
                indices.append(next_index)
                next_index += 1
            else:
                indices.append(n_active)
        self._channelindex = jnp.array(indices)

    def __call__(self):
        """What to run when the TransientArraySpectrum is called"""
        padded = jnp.concatenate((self.data, jnp.zeros(1, dtype=self.data.dtype)))
        return padded[self._channelindex]

    @property
    def shape(self):
        """The shape of the spectrum as seen by the model frame, i.e. including
        the (zero-valued) quiescent channels"""
        return (len(self._channelindex),)
