import equinox as eqx
import jax.numpy as jnp

from .bbox import Box
from .module import Module


class Spectrum(Module):
    bbox: Box = eqx.field(static=True, init=False)


class ArraySpectrum(Spectrum):
    data: jnp.array

    def __init__(self, data):
        self.data = data
        super().__post_init__()
        self.bbox = Box(self.data.shape)

    def __call__(self):
        return self.data

class StaticArraySpectrum(Spectrum):
    data: jnp.array
    channelindex: jnp.ndarray

    def __init__(self, data, filters, scene):
        self.data = data
        super().__post_init__()
        self.channelindex = jnp.array([filters.index(c[0]) for c in scene.frame.channels])
        self.bbox = Box([len(self.channelindex)])
        
    def __call__(self):
        return self.data[self.channelindex]


    
