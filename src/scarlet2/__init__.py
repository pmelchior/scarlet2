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
from .infer import fit, sample
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
from .validation import check_fit, check_observation, check_scene, check_source
from .validation_utils import VALIDATION_SWITCH, set_validation
from .wavelets import Starlet

# for * imports and docs
__all__ = [
    "constraint",
    "detect",
    "init",
    "measure",
    "plot",
    "fit",
    "sample",
    "Scenery",
    "Parameterization",
    "Box",
    "Frame",
    "Parameter",
    "Parameters",
    "Module",
    "Morphology",
    "ProfileMorphology",
    "GaussianMorphology",
    "SersicMorphology",
    "StarletMorphology",
    "Observation",
    "CorrelatedObservation",
    "PSF",
    "ArrayPSF",
    "GaussianPSF",
    "Scene",
    "Component",
    "DustComponent",
    "Source",
    "PointSource",
    "Spectrum",
    "StaticArraySpectrum",
    "TransientArraySpectrum",
    "Starlet",
    "check_fit",
    "check_observation",
    "check_scene",
    "check_source",
    "relative_step",
    "set_validation",
    "VALIDATION_SWITCH",
]
