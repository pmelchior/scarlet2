import copy
from jax import numpy as jnp
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

class DustySource(Module):
    spectrum: Spectrum
    morphology: Morphology
    host_spectrum: Spectrum
    host_morphology: Morphology

    def __init__(self, center, spectrum, morphology, host_spectrum, host_morphology):
        self.spectrum = spectrum
        self.morphology = morphology
        self.morphology.center_bbox(center)

        self.host_spectrum = host_spectrum
        self.host_morphology = host_morphology
        self.host_morphology.center_bbox(center)

        super().__post_init__()

        # add this source to the active scene
        try:
            Scenery.scene.sources.append(self)
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

    def get_attenuation(self):
        temp = 1-self.morphology()[None, :, :]
        temp = jnp.where(temp<0,0,temp)
        Ag = jnp.log10(temp) / (-0.4)
        Kg = self.spectrum()[0]
        EBV = Ag/Kg

        return 10**( -0.4 * EBV * self.spectrum()[:, None, None] )
        # return 10**( -0.4 * self.morphology()[None, :, :] * self.spectrum()[:, None, None] )
    
    def __call__(self):
        original_model = self.host_spectrum()[:, None, None] * self.host_morphology()[None, :, :]
        dusty_model = self.get_attenuation()
        return original_model * dusty_model
    
    @property
    def bbox(self):
        return self.spectrum.bbox @ self.morphology.bbox
