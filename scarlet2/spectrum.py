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

    In the frames of transient scenes, the attribute :py:attr:`~scarlet2.Frame.channels` are overloaded and defined
    with a spectral and a temporal component, e.g. `channel = (band, epoch)`.
    This class is for models that do not vary in time, i.e. only have a spectral dependency.
    The length of :py:attr:`data` is thus given by the number of distinct spectral bands.
    """
    data: jnp.array
    """Data to describe the static spectrum
    
    The order in this array should be given by :py:attr:`bands`.
    """
    bands: list
    """Identifier for the list of unique bands in the model frame channels"""
    _channelindex: jnp.array = eqx.field(static=True)

    def __init__(self, data, bands, band_selector=lambda channel: channel[0]):
        """
        Parameters
        ----------
        data: array
            Spectrum without temporal variation. Contains as many elements as there are spectral channels in the model.
        bands: list, array
            Identifier for the list of unique bands in the model frame channels
        band_selector: callable, optional
            Identify the spectral "band" component from the name/ID used in the channels of the model frame

        Examples
        --------
        >>> # model channels: [('G',0),('G',1),('R',0),('R',1),('R',2)]
        >>> spectrum = jnp.ones(2)
        >>> bands = ['G','R']
        >>> band_selector = lambda channel: channel[0]
        >>> StaticArraySpectrum(spectrum, bands, band_selector=band_selector)

        This constructs a 2-element spectrum to describe the spectral properties in all epochs 0,1,2.

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

        self.data = data
        self.bands = bands
        self._channelindex = jnp.array([self.bands.index(band_selector(c)) for c in frame.channels])

    def __call__(self):
        return self.data[self._channelindex]

    @property
    def shape(self):
        return len(self.channelindex),


class TransientArraySpectrum(Spectrum):
    """Variable source in a transient scene with possible quiescent periods

    In the frames of transient scenes, the attribute :py:attr:`~scarlet2.Frame.channels` are overloaded and defined
    with a spectral and a temporal component, e.g. `channel = (band, epoch)`.
    This class is for models that vary in time, especially if they have periods of inactivity.
    The length of :py:attr:`data` is given by the number channels in the model frame, but during inactive epochs, the
    emission is set to zero.
    """
    data: jnp.array
    """Data to describe the variable spectrum. 
    
    The length of this vector is identical to the number of channels in the model frame.
    """
    epochs: list
    """Identifier for the list of active epochs. If set to `None`, all epochs are considered active"""
    _epochmultiplier: jnp.array = eqx.field(static=True)

    def __init__(self, data, epochs=None, epoch_selector=lambda channel: channel[1]):
        """
        Parameters
        ----------
        data: array
            Spectrum array. Contains as many elements as there are spectro-temporal channels in the model.
        epochs: list, array, optional
            List of temporal "epoch" identifiers for the active phases of the source.
        epoch_selector: callable, optional
            Identify the temporal "epoch" component from the name/ID used in the channels of the model frame

        Examples
        --------
        >>> # model channels: [('G',0),('G',1),('R',0),('R',1),('R',2)]
        >>> spectrum = jnp.ones(5)
        >>> epochs = [0, 1]
        >>> epoch_selector = lambda channel: channel[1]
        >>> TransientArraySpectrum(spectrum, epochs, epoch_selector=epoch_selector)

        This sets the spectrum to active during epochs 0 and 1, and mask the spectrum element for `('R',2)` with zero.

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
        self.data = data
        self.epochs = epochs
        self._epochmultiplier = jnp.array([1.0 if epoch_selector(c) in epochs else 0.0 for c in frame.channels])

    def __call__(self):
        return jnp.multiply(self.data, self._epochmultiplier)

    @property
    def shape(self):
        return self.data.shape
