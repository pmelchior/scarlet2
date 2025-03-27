"""Main namespace for scarlet2"""


class Scenery:
    """Class to hold the context for the current scene"""
    scene = None


from . import init
from . import measure
from . import plot
from .bbox import Box
from .frame import Frame
from .module import Parameter, Parameters, Module, relative_step
from .morphology import Morphology, ArrayMorphology, ProfileMorphology, GaussianMorphology, SersicMorphology, \
    StarletMorphology
from .observation import Observation
from .psf import PSF, ArrayPSF, GaussianPSF
from .scene import Scene
from .source import Component, DustComponent, Source, PointSource
from .spectrum import Spectrum, ArraySpectrum, StaticArraySpectrum, TransientArraySpectrum
from .wavelets import Starlet

# for * imports and docs
__all__ = [item for item in dir() if item[0:2] != '__']  # remove dunder
