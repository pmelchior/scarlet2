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


from . import init, measure, plot
from .bbox import Box
from .frame import Frame
from .module import Module, Parameter, Parameters, relative_step
from .morphology import (
    GaussianMorphology,
    Morphology,
    ProfileMorphology,
    SersicMorphology,
    StarletMorphology,
)
from .observation import Observation
from .psf import PSF, ArrayPSF, GaussianPSF
from .scene import Scene
from .source import Component, DustComponent, PointSource, Source
from .spectrum import Spectrum, StaticArraySpectrum, TransientArraySpectrum
from .validation import check_fit, check_observation, check_scene, check_source
from .validation_utils import VALIDATION_SWITCH, set_validation
from .wavelets import Starlet

# for * imports and docs
__all__ = [
    "init",
    "measure",
    "plot",
    "Scenery",
    "Box",
    "Frame",
    "Parameter",
    "Parameters",
    "Module",
    "relative_step",
    "Morphology",
    "ProfileMorphology",
    "GaussianMorphology",
    "SersicMorphology",
    "StarletMorphology",
    "Observation",
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
    "VALIDATION_SWITCH",
    "set_validation",
]
