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


class ArraySpectrum(Spectrum):
    data: jnp.array
    """1D spectrum"""

    def __init__(self, data):
        """Spectrum defined by a 1D array"""
        self.data = data

    def __call__(self):
        return self.data

    @property
    def shape(self):
        return self.data.shape


# TODO: why is this derived from Spectrum instead of ArraySpectrum?
class StaticArraySpectrum(Spectrum):
    data: jnp.array
    channelindex: list = eqx.field(static=True)
    """TODO"""

    def __init__(self, data, filters):
        """TODO"""
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

        self.data = data
        # TODO: why c[0]? Should this not be the entire channel name?
        self.channelindex = jnp.array([filters.index(c[0]) for c in frame.channels])

    def __call__(self):
        return self.data[self.channelindex]

    @property
    def shape(self):
        return len(self.channelindex),


# TODO: why is this derived from Spectrum instead of ArraySpectrum?
class TransientArraySpectrum(Spectrum):
    data: jnp.array
    _epochmultiplier: jnp.array = eqx.field(static=True)
    """TODO"""

    def __init__(self, data, epochs):
        """TODO"""
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise
        self.data = data
        self._epochmultiplier = jnp.array([1.0 if c in epochs else 0.0 for c in frame.channels])

    def __call__(self):
        return jnp.multiply(self.data, self._epochmultiplier)

    @property
    def shape(self):
        return self.data.shape
