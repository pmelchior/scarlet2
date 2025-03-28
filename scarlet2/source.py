import operator

import equinox as eqx
import jax
import jax.numpy as jnp
from astropy.coordinates import SkyCoord

from . import Scenery
from .bbox import Box, overlap_slices
from .module import Module
from .morphology import Morphology
from .spectrum import Spectrum


class Component(Module):
    """Single component of a hyperspectral model

    The parameterization of the 3D model (channel, height, width) is defined by the outer product of
    `spectrum` and `morphology`. That means that there is no variation of the spectrum in spatial direction.
    The `center` coordinate is only needed to define the bounding box and place the component in the model frame.
    """
    center: jnp.ndarray
    """Center position, in pixel coordinates of the model frame"""
    spectrum: Spectrum
    """Spectrum model"""
    morphology: Morphology
    """Morphology model"""
    bbox: Box = eqx.field(static=True, init=False)
    """Bounding box of the model, in pixel coordinates of the model frame"""

    def __init__(self, center, spectrum, morphology):
        """
        Parameters
        ----------
        center: array, :py:class:`astropy.coordinates.SkyCoord`
            Center position. If given as astropy sky coordinate, it will be transformed with the WCS of the model frame.
        spectrum: :py:class:`~scarlet2.Spectrum`
        morphology: :py:class:`~scarlet2.Morphology`

        Examples
        --------
        To uniquely determine coordinates, the creation of components is restricted to a context defined by
        a :py:class:`~scarlet2.Scene`, which define the :py:class:`~scarlet2.Frame` of the model.

        >>> with Scene(model_frame) as scene:
        >>>    component = Component(center, spectrum, morphology)
        """
        assert isinstance(spectrum, Spectrum)
        self.spectrum = spectrum
        assert isinstance(morphology, Morphology)
        self.morphology = morphology

        if isinstance(center, SkyCoord):
            try:
                center = Scenery.scene.frame.get_pixel(center)
            except AttributeError:
                print("`center` defined in sky coordinates can only be created within the context of a Scene")
                print("Use 'with Scene(frame) as scene: (...)'")
                raise
        self.center = center

        box = Box(spectrum.shape)
        box2d = Box(morphology.shape)
        box2d.set_center(center.astype(int))
        self.bbox = box @ box2d

    def __call__(self):
        # Boxed and centered model
        delta_center = (self.bbox.center[-2] - self.center[-2], self.bbox.center[-1] - self.center[-1])
        return self.spectrum()[:, None, None] * self.morphology(delta_center=delta_center)[None, :, :]


class DustComponent(Component):
    """Component with negative exponential model

    This component is meant to describe the dust attenuation, :math:`\\exp(-\\tau)`,
    where :math:`\\tau` is the hyperspectral model defined by the base :py:class:`~scarlet2.Component`.
    """

    def __call__(self):
        return jnp.exp(-super().__call__())


class Source(Component):
    """Source model

    The class is the basic parameterization for sources in :py:class:`~scarlet2.Scene`.
    """
    components: list
    """List of components in this source"""
    component_ops: list = eqx.field(static=True)
    """List of operators to combine `components` for the final model"""

    def __init__(self, center, spectrum, morphology):
        """
        Parameters
        ----------
        center: array, :py:class:`astropy.coordinates.SkyCoord`
            Center position. If given as astropy sky coordinate, it will be transformed with the WCS of the model frame.
        spectrum: :py:class:`~scarlet2.Spectrum`
        morphology: :py:class:`~scarlet2.Morphology`

        Examples
        --------
        A source  declaration is restricted to a context of a :py:class:`~scarlet2.Scene`,
        which defines the :py:class:`~scarlet2.Frame` of the entire model.

        >>> with Scene(model_frame) as scene:
        >>>    source = Source(center, spectrum, morphology)

        A source can comprise one or multiple :py:class:`~scarlet2.Component`, which can be added by
        :py:func:`add_component` or operators `+=` (for an additive component) or `*=` (for a multiplicative component).

        >>> with Scene(model_frame) as scene:
        >>>    source = Source(center, spectrum, morphology)
        >>>    source *= DustComponent(center, dust_spectrum, dust_morphology)
        """
        # set the base component
        super().__init__(center, spectrum, morphology)
        # create the empty component list
        self.components = list()
        self.component_ops = list()

        # add this source to the active scene
        try:
            Scenery.scene.sources.append(self)
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

    def add_component(self, component, op):
        """Add `component` to this source

        Parameters
        ----------
        component: :py:class:`~scarlet2.Component`
        op: callable
            Operator to combine this `component` with those before it in the list :py:attr:`components`.
            Conventional operators from the :py:mod:`operator` package can be used.
            Signature: op(x,y) -> z, where all terms have the same shapes
        """
        assert isinstance(component, (Source, Component))

        # if component is a source, it's already registered in scene
        # remove it from scene to not call it twice
        if isinstance(component, Source):
            try:
                Scenery.scene.sources.remove(component)
            except AttributeError:
                print("Source can only be modified within the context of a Scene")
                print("Use 'with Scene(frame) as scene: Source(...)'")
                raise
            except ValueError:
                pass

        # adding a full source will maintain its ownership of components:
        # hierarchical definition of sources withing sources
        self.components.append(component)
        self.component_ops.append(op)
        return self

    def __iadd__(self, component):
        return self.add_component(component, operator.add)

    def __imul__(self, component):
        return self.add_component(component, operator.mul)

    def __call__(self):
        base = super()
        model = base.__call__()
        for component, op in zip(self.components, self.component_ops):
            model_ = component()
            # cut out regions from model and model_
            bbox, bbox_ = overlap_slices(base.bbox, component.bbox, return_boxes=True)
            sub_model = jax.lax.dynamic_slice(model, bbox.start, bbox.shape)
            sub_model_ = jax.lax.dynamic_slice(model_, bbox_.start, bbox_.shape)
            # combine with operator
            sub_model = op(sub_model, sub_model_)
            # add model_ back in full model
            model = jax.lax.dynamic_update_slice(model, sub_model, bbox.start)
        return model


class PointSource(Source):
    def __init__(self, center, spectrum):
        """Model for point sources

        Because the morphology is determined by the model PSF, it does not need to be provided.

        Parameters
        ----------
        center: jnp.array, :py:class:`astropy.coordinates.SkyCoord`
            Center position. If given as astropy sky coordinate, it will be transformed with the WCS of the model frame.
        spectrum: :py:class:`~scarlet2.Spectrum`

        Examples
        --------
        A source declaration is restricted to a context of a :py:class:`~scarlet2.Scene`,
        which defines the :py:class:`~scarlet2.Frame` of the entire model.

        >>> with Scene(model_frame) as scene:
        >>>    point_source = PointSource(center, spectrum)
        """
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise
        if frame.psf is None:
            raise AttributeError("PointSource can only be create with a PSF in the model frame")
        morphology = frame.psf.morphology

        super().__init__(center, spectrum, morphology)
