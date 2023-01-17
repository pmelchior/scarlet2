from .module import Module, Parameter


class PSF(Module):
    def __call__(self):
        raise NotImplementedError


class ArrayPSF(Parameter, PSF):
    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)
