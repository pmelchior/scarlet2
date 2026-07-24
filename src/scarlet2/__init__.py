#  ruff: noqa: E402
"""Main namespace for scarlet2"""


class Scenery:
    """Class to hold the context for the current scene

    See Also
    --------
    :py:class:`~scarlet2.Scene`
    """

    scene = None
    """Scene of the currently opened context"""


class Parameterization:
    """Class to hold the context for the current parameter set

    See Also
    --------
    :py:class:`~scarlet2.Parameters`
    """

    parameters = None
    """Parameters of the currently opened context"""


parameter_registry = dict()


from . import constraint, detect, init, measure, plot

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from pathlib import Path

__citation__ = __bibtex__ = (Path(__file__).parent / "citation.bib").read_text()

from .bbox import Box
from .frame import Frame
from .infer import PairSimilarity
from .module import Module, Parameter, Parameters, relative_step
from .morphology import (
    GaussianMorphology,
    Morphology,
    ProfileMorphology,
    SersicMorphology,
    StarletMorphology,
)
from .observation import CorrelatedObservation, Observation
from .psf import PSF, ArrayPSF, GaussianPSF
from .scene import Scene
from .source import Component, DustComponent, PointSource, Source
from .spectrum import Spectrum, StaticArraySpectrum, TransientArraySpectrum
from .validation_utils import set_validation
from .wavelets import Starlet

# for * imports and docs
__all__ = [
    "constraint",
    "detect",
    "infer",
    "init",
    "measure",
    "plot",
    "validation",
    "ArrayPSF",
    "Box",
    "Component",
    "CorrelatedObservation",
    "DustComponent",
    "Frame",
    "GaussianMorphology",
    "GaussianPSF",
    "Module",
    "Morphology",
    "Observation",
    "PairSimilarity",
    "Parameter",
    "Parameterization",
    "Parameters",
    "PointSource",
    "ProfileMorphology",
    "PSF",
    "Scene",
    "Scenery",
    "SersicMorphology",
    "Source",
    "Spectrum",
    "Starlet",
    "StarletMorphology",
    "StaticArraySpectrum",
    "TransientArraySpectrum",
    "relative_step",
    "set_validation",
]
