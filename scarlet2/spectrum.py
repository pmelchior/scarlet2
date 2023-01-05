import equinox as eqx
from .bbox import Box
from .module import Parameter, Module

class Spectrum(Module):
    bbox: Box = eqx.static_field()


class ArraySpectrum(Spectrum, Parameter):
    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)
        self.bbox = Box(self.value.shape)