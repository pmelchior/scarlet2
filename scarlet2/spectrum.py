import equinox as eqx
from .bbox import Box
from .module import Parameter, Module
import jax.numpy as jnp

class Spectrum(Module):
    bbox: Box = eqx.static_field()

class Specind(Module):
    bbox: Box = eqx.static_field()

class ArraySpectrum(Spectrum, Parameter):
    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)
        self.bbox = Box(self.value.shape)

class StaticArraySpectrum(Spectrum, Parameter):
    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)  
        self.bbox = Box(jnp.repeat(self.value,self.repeats).shape)

