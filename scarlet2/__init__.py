class Scenery:
    # static store for context manager
    scene = None


from . import initialization as init
from . import measure
from . import plot
from .bbox import Box
from .frame import Frame
from .module import Parameter, Module, relative_step
from .morphology import Morphology, ArrayMorphology, GaussianMorphology, SersicMorphology, StarletMorphology
from .observation import Observation
from .psf import PSF, ArrayPSF, GaussianPSF
from .scene import Scene
from .source import Component, DustComponent, Source, PointSource
from .spectrum import Spectrum, ArraySpectrum, StaticArraySpectrum, TransientArraySpectrum
from .wavelets import Starlet
