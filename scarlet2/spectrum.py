import equinox as eqx
import jax.numpy as jnp

from . import Scenery
from .module import Module


class Spectrum(Module):

    @property
    def shape(self):
        raise NotImplementedError


class ArraySpectrum(Spectrum):
    data: jnp.array

    def __init__(self, data):
        self.data = data

    def __call__(self):
        return self.data

    @property
    def shape(self):
        return self.data.shape


class StaticArraySpectrum(Spectrum):
    data: jnp.array
    channelindex: list = eqx.field(static=True)

    def __init__(self, data, filters):
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

        self.data = data
        self.channelindex = jnp.array([filters.index(c[0]) for c in frame.channels])

    def __call__(self):
        return self.data[self.channelindex]

    @property
    def shape(self):
        return len(self.channelindex),


class TransientArraySpectrum(Spectrum):
    data: jnp.array
    _epochmultiplier: jnp.array = eqx.field(static=True)

    def __init__(self, data, epochs):
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
