import copy

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
        if not isinstance(center, Parameter):
            center = Parameter(center, fixed=True)
        self.morphology.set_center(center)

        # add this source to the active scene
        try:
            Scenery.scene.sources.append(self)
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

    def __call__(self):
        # Boxed model
        return self.spectrum()[:, None, None] * self.morphology()[None, :, :]

    @property
    def bbox(self):
        return self.spectrum.bbox @ self.morphology.bbox


class PointSource(Source):
    def __init__(self, center, spectrum):
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise
        if frame.psf is None:
            raise AttributeError("PointSource can only be create with a PSF in the model frame")

        morphology = copy.deepcopy(frame.psf.morphology)
        super().__init__(center, spectrum, morphology)
