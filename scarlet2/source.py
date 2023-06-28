from .module import Module, Parameter
from .morphology import Morphology, GaussianMorphology
from .scene import Scenery
from .spectrum import Spectrum,Specind
import jax.numpy as jnp

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
            print("Sources can only be created within the context of a Scene")
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
    specind: Specind
    def __init__(self, center, spectrum, morphology, repeats):
        self.spectrum = spectrum
        self.morphology = morphology
        if not isinstance(center, Parameter):
            center = Parameter(center, fixed=True)
        self.morphology.set_center(center) 
        self.specind = jnp.repeat(jnp.arange(spectrum.value.shape[0]),repeats)
        
        # add this source to the active scene
        try:
            Scenery.scene.sources.append(self)
        except AttributeError:
            print("Sources can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

    def __call__(self):
        # Boxed model
        return self.spectrum()[self.specind][:, None, None] * self.morphology()[None, :, :]

    @property
    def bbox(self): 
        return self.spectrum.bbox @ self.morphology.bbox



class PointSource(Source):
    def __init__(self, center, spectrum):
        try:
            sigma = Scenery.scene.frame.psf.sigma
        except AttributeError:
            print("Sources can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise
        morphology = GaussianMorphology(center, sigma)
        super().__init__(center, spectrum, morphology)
