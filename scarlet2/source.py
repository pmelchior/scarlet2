import jax

from .module import Module, Parameter
from .morphology import Morphology
from .scene import Scenery
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

        # add this source to the active scene
        try:
            Scenery.scene.sources.append(self)
        except AttributeError:
            print("Sources can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

    def __call__(self):
        # Boxed model
        return self.spectrum()[:, None, None] * self.morphology()[None, :, :]

    @property
    def bbox(self):
        return self.spectrum.bbox @ self.morphology.bbox