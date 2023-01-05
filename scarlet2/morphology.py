import equinox as eqx

from .bbox import Box
from .module import Parameter, Module


class Morphology(Module):
    bbox: Box = eqx.static_field()
    center: (Parameter, None) = None

    def set_center(self, center):
        object.__setattr__(self, 'center', center)
        center_ = tuple(_.item() for _ in center.value.astype(int))
        self.bbox.set_center(center_)


class ArrayMorphology(Morphology, Parameter):
    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)
        self.bbox = Box(self.value.shape)
