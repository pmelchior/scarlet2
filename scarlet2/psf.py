import jax.numpy as jnp

from .module import Module, Parameter
from .morphology import GaussianMorphology


class PSF(Module):
    def __call__(self):
        raise NotImplementedError


class ArrayPSF(Parameter, PSF):
    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)


class GaussianPSF(PSF):
    morphology: GaussianMorphology

    def __init__(self, sigma):
        self.morphology = GaussianMorphology(jnp.zeros(2), sigma)

    def __call__(self):
        return self.morphology()
