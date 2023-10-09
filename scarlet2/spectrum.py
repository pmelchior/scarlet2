import equinox as eqx
import jax.numpy as jnp

from .bbox import Box
from .module import Module
from .scene import Scenery


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
        self.bbox = Box([len(self.channelindex)])
        super().__post_init__()
    
    def __call__(self):
        return self.data[self.channelindex]

class SMCAttenuationSpectrum(Spectrum):
    # https://ui.adsabs.harvard.edu/abs/2003AJ....126.1131R/abstract
    # http://www.bo.astro.it/~micol/Hyperz/old_public_v1/hyperz_manual1/node10.html

    channel_wavelengths : jnp.array # angstrom

    def __init__(self, channel_wavelengths):
        self.channel_wavelengths = channel_wavelengths * 1e-4 # micrometer

        super().__post_init__()
        self.bbox = Box(self.channel_wavelengths.shape)

    def __call__(self):
        A_lambda = 1.39 * self.channel_wavelengths **-1.2

        return A_lambda
