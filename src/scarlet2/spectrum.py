import equinox as eqx
import jax.numpy as jnp

from . import Scenery
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
    are overloaded and defined with a spectral and a temporal component, e.g.
    `channel = (band, epoch)`.
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

    def __init__(self, data, bands, band_selector=lambda channel: channel[0]):
        """
        Parameters
        ----------
        data: array
            Spectrum without temporal variation. Contains as many elements as there
            are spectral channels in the model.
        bands: list, array
            Identifier for the list of unique bands in the model frame channels
        band_selector: callable, optional
            Identify the spectral "band" component from the name/ID used in the
            channels of the model frame

        Examples
        --------
        >>> # model channels: [('G',0),('G',1),('R',0),('R',1),('R',2)]
        >>> spectrum = jnp.ones(2)
        >>> bands = ['G','R']
        >>> band_selector = lambda channel: channel[0]
        >>> StaticArraySpectrum(spectrum, bands, band_selector=band_selector)

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

        assert len(data) == len(bands)
        self.data = data
        self.bands = bands
        self._channelindex = jnp.array([self.bands.index(band_selector(c)) for c in frame.channels])

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
    are overloaded and defined with a spectral and a temporal component, e.g.
    `channel = (band, epoch)`. This class is for models that vary in time, especially
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

    def __init__(self, data, epochs=None, epoch_selector=lambda channel: channel[1]):
        """
        Parameters
        ----------
        data: array
            Spectrum array, restricted to the active epochs. Contains as many elements
            as there are spectro-temporal channels in the model for which
            `epoch_selector(channel)` is in `epochs`.
        epochs: list, array, optional
            List of temporal "epoch" identifiers for the active phases of the source.
            If `None`, all epochs are considered active.
        epoch_selector: callable, optional
            Identify the temporal "epoch" component from the name/ID used in the
            channels of the model frame

        Examples
        --------
        >>> # model channels: [('G',0),('G',1),('R',0),('R',1),('R',2)]
        >>> spectrum = jnp.ones(4)  # one value per active (band, epoch) channel
        >>> epochs = [0, 1]
        >>> epoch_selector = lambda channel: channel[1]
        >>> TransientArraySpectrum(spectrum, epochs, epoch_selector=epoch_selector)

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
        is_active = [epochs is None or epoch_selector(c) in epochs for c in frame.channels]
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
