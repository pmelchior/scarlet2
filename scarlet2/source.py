import jax

from .module import Module, Parameter
from .morphology import Morphology
from .spectrum import Spectrum


class Source(Module):
    spectrum: Spectrum
    morphology: Morphology

    def __init__(self, center, spectrum, morphology):
        self.spectrum = spectrum
        self.morphology = morphology
        if isinstance(center, jax.numpy.ndarray):
            center = Parameter(center)
        self.morphology.set_center(center)

    def __call__(self):
        # Boxed model
        return self.spectrum()[:, None, None] * self.morphology()[None, :, :]

    @property
    def bbox(self):
        return self.spectrum.bbox @ self.morphology.bbox