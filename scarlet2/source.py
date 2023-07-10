import copy

from .module import Module
from .morphology import Morphology
from .scene import Scenery
from .spectrum import Spectrum


class Source(Module):
    spectrum: Spectrum
    morphology: Morphology

    def __init__(self, center, spectrum, morphology):
        self.spectrum = spectrum
        self.morphology = morphology
        self.morphology.center_bbox(center)
        super().__post_init__()

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

class StaticSource(Module):
    spectrum: Spectrum
    morphology: Morphology
    def __init__(self, center, spectrum, morphology):
        self.spectrum = spectrum
        self.morphology = morphology
        self.morphology.center_bbox(center)
        super().__post_init__() 

        # add this source to the active scene
        try:
            Scenery.scene.sources.append(self)
        except AttributeError:
            print("Sources can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

    def __call__(self):
        # Boxed model
        return self.spectrum()[self.spectrum.channelindex, None, None] * self.morphology()[None, :, :]

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

        # use frame's PSF but with free center parameter
        morphology = copy.deepcopy(frame.psf.morphology)
        object.__setattr__(morphology, 'center', center)
        super().__init__(center, spectrum, morphology)
