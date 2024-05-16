import jax.numpy as jnp

from .module import Module
from .morphology import GaussianMorphology


class PSF(Module):
    pass


class ArrayPSF(PSF):
    morphology: jnp.ndarray

    def __call__(self):
        return self.morphology / self.morphology.sum()


class GaussianPSF(PSF):
    morphology: GaussianMorphology

    def __init__(self, sigma):
        self.morphology = GaussianMorphology(jnp.zeros(2), sigma)

    def __call__(self):
        morph = self.morphology()
        morph /= morph.sum()
        return morph
