import functools
import operator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import Scenery
from .bbox import Box, overlap_slices
from .channel import EPOCH
from .module import Module, _to_pixels
from .morphology import Morphology
from .spectrum import Spectrum
from .validation_utils import (
    ValidationMethodCollector,
)


class Component(Module):
    """Single component of a hyperspectral model

    The parameterization of the 3D model (channel, height, width) is defined by
    the outer product of `spectrum` and `morphology`. That means that there is no
    variation of the spectrum in spatial direction. The `center` coordinate is only
    needed to define the bounding box and place the component in the model frame.
    """

    center: jnp.ndarray
    """Center position, in sky coordinates or pixel coordinates of the model frame"""
    spectrum: (jnp.array, Spectrum)
    """Spectrum model"""
    morphology: (jnp.array, Morphology)
    """Morphology model"""
    bbox: Box
    """Bounding box of the model, in sky coordinates or pixel coordinates of the model frame"""

    def __init__(self, center, spectrum, morphology, bbox=None):
        """
        Parameters
        ----------
        center: array, :py:class:`astropy.coordinates.SkyCoord`
            Center position. If given as astropy sky coordinate, it will be
            transformed with the WCS of the model frame.
        spectrum: :py:class:`~scarlet2.Spectrum`
            The spectrum of the component.
        morphology: :py:class:`~scarlet2.Morphology`
            The morphology of the component.
        bbox: :py:class:`~scarlet2.Box`, optional
            Bounding box of the component in pixel coordinates of the model frame.

        Examples
        --------
        To uniquely determine coordinates, the creation of components is restricted
        to a context defined by a :py:class:`~scarlet2.Scene`, which define the
        :py:class:`~scarlet2.Frame` of the model.

        >>> with Scene(model_frame) as scene:
        >>>    component = Component(center, spectrum, morphology)
        """
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Attributes defined in sky coordinates can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: (...)'")
            raise

        self.center = _to_pixels(frame, center)
        self.spectrum = spectrum  # no unit conversion on spectrum (not yet defined)
        self.morphology = _to_pixels(frame, morphology)

        # define box with integer pixel coordinates to place the component
        if bbox is None:
            box = Box(spectrum.shape)
            box2d = Box(morphology.shape)
            box2d.set_center(self.center.astype(int))
            bbox = box @ box2d
        self.bbox = bbox

    def __call__(self):
        """What to run when Component is called"""
        # Boxed and centered model
        delta_center = (self.center[-2] - self.bbox.center[-2], self.center[-1] - self.bbox.center[-1])
        spectrum = self.spectrum() if isinstance(self.spectrum, Module) else self.spectrum
        morph = (
            self.morphology(delta_center=delta_center)
            if isinstance(self.morphology, Module)
            else self.morphology
        )
        return spectrum[:, None, None] * morph[None, :, :]


class DustComponent(Component):
    """Component with negative exponential model

    This component is meant to describe the dust attenuation, :math:`\\exp(-\\tau)`,
    where :math:`\\tau` is the hyperspectral model defined by the base :py:class:`~scarlet2.Component`.
    """

    def __call__(self):
        """What to run when DustComponent is called"""
        return jnp.exp(-super().__call__())


class MovingComponent(Component):
    """Component that moves between epochs

    The model has the same morphology in every channel and a free spectrum amplitude
    in every channel, but a different center in every epoch. Each channel thus
    contributes one 2D slice, `spectrum[c] * morphology`, placed at the center of
    its epoch. The slices are stacked into a 3D model, whose spatial bounding box
    is the union of the boxes of all epochs.

    Without `epochs`, every channel has its own center. With `epochs`, the channels are
    assigned to centers by the `EPOCH` component of the model frame
    :py:attr:`~scarlet2.Frame.channels`, so that all channels of the same epoch share
    the same center.
    """

    epochs: (tuple, None) = eqx.field(static=True)
    """Identifiers of the epochs, in the order of :py:attr:`center`, or `None` if every
    channel has its own center"""
    _epochindex: tuple = eqx.field(static=True, repr=False)
    _origins: tuple = eqx.field(static=True, repr=False)
    _box_centers: tuple = eqx.field(static=True, repr=False)

    def __init__(self, center, spectrum, morphology, bbox=None, epochs=None):
        """
        Parameters
        ----------
        center: array, :py:class:`astropy.coordinates.SkyCoord`
            Center positions, one for every channel (if `epochs` is `None`) or one for every
            epoch in `epochs`, with shape `(N, 2)` in pixel coordinates of the model frame,
            or as an array of astropy sky coordinates.
        spectrum: array, :py:class:`~scarlet2.Spectrum`
            The spectrum of the component, with one element for every channel of the model frame.
        morphology: array, :py:class:`~scarlet2.Morphology`
            The morphology of the component, the same in all epochs.
        bbox: :py:class:`~scarlet2.Box`, optional
            Ignored, the bounding box is determined from the centers of all epochs.
        epochs: list, optional
            Identifiers of the epochs, in the order of `center`. Every epoch in the `EPOCH`
            component of the model frame channels needs to be listed. If `None`, every
            channel has its own center.

        Examples
        --------
        >>> from scarlet2 import BAND, EPOCH, Channels, Frame, GaussianMorphology, Scene
        >>> from scarlet2.bbox import Box
        >>> channels = Channels([("g", 0), ("r", 1)], (BAND, EPOCH))
        >>> frame = Frame(Box((2, 30, 30)), channels=channels)
        >>> with Scene(frame):
        ...     component = MovingComponent(
        ...         jnp.array([[10., 10.], [15., 20.]]), jnp.ones(2), GaussianMorphology(1.0)
        ...     )
        >>> component.bbox
        Box(shape=(2, 20, 25), origin=(0, 3, 3))
        """
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Attributes defined in sky coordinates can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: (...)'")
            raise

        self.center = jnp.asarray(_to_pixels(frame, center), dtype=float)
        self.spectrum = spectrum
        self.morphology = _to_pixels(frame, morphology)

        n_channels = len(frame.channels)
        if spectrum.shape != (n_channels,):
            raise ValueError(f"Spectrum shape {spectrum.shape} does not match {n_channels} model channels")

        if epochs is None:
            # one center per channel
            n_centers = n_channels
            self.epochs = None
            self._epochindex = tuple(range(n_channels))
        else:
            channel_epochs = frame.channels.get_component(EPOCH)
            if channel_epochs is None:
                raise ValueError(f"Channel format {frame.channels.format!r} of the model frame has no EPOCH")
            epochs = list(epochs)
            missing = [epoch for epoch in channel_epochs if epoch not in epochs]
            if missing:
                raise ValueError(f"Epochs {missing} of the model frame channels are not in {epochs}")
            n_centers = len(epochs)
            self.epochs = tuple(epochs)
            self._epochindex = tuple(epochs.index(epoch) for epoch in channel_epochs)
        if self.center.shape != (n_centers, 2):
            raise ValueError(f"Center needs shape ({n_centers}, 2), got {self.center.shape}")

        # integer box of every epoch, same as for Component
        boxes = []
        for center_ in self.center:
            box2d = Box(self.morphology.shape)
            box2d.set_center(center_.astype(int))
            boxes.append(box2d)
        box2d = functools.reduce(operator.or_, boxes)
        self.bbox = Box((n_channels,)) @ box2d
        self._origins = tuple(
            tuple(o - o_ for o, o_ in zip(b.origin, box2d.origin, strict=True)) for b in boxes
        )
        self._box_centers = tuple(b.center for b in boxes)

    def __call__(self):
        """What to run when MovingComponent is called"""
        delta_center = self.center - jnp.array(self._box_centers, dtype=float)
        spectrum = self.spectrum() if isinstance(self.spectrum, Module) else self.spectrum
        if isinstance(self.morphology, Module):
            morphs = jax.vmap(lambda d: self.morphology(delta_center=d))(delta_center)
        else:
            morphs = jnp.broadcast_to(self.morphology, (len(self.center), *self.morphology.shape))

        # one 2D slice per channel, from the morphology of its epoch
        epochindex = np.array(self._epochindex)
        values = spectrum[:, None, None] * morphs[epochindex]

        # place every slice at the box of its epoch
        h, w = morphs.shape[-2:]
        origins = np.array(self._origins)[epochindex]  # (C, 2)
        rows = origins[:, 0, None] + np.arange(h)  # (C, h)
        cols = origins[:, 1, None] + np.arange(w)  # (C, w)
        channels = np.arange(len(epochindex))
        model = jnp.zeros(self.bbox.shape)
        return model.at[channels[:, None, None], rows[:, :, None], cols[:, None, :]].set(values)


class Source(Component):
    """Source model

    The class is the basic parameterization for sources in :py:class:`~scarlet2.Scene`.
    """

    components: list
    """List of components in this source"""
    component_ops: list
    """List of operators to combine `components` for the final model"""

    def __init__(self, center, spectrum, morphology, bbox=None, **kwargs):
        """
        Parameters
        ----------
        center: array, :py:class:`astropy.coordinates.SkyCoord`
            Center position in sky coordinates or pixel coordinates of the model frame.
        spectrum: array, :py:class:`~scarlet2.Spectrum`
            The spectrum of the source.
        morphology: array, :py:class:`~scarlet2.Morphology`
            The morphology of the source.
        bbox: :py:class:`~scarlet2.Box`, optional
            Bounding box of the component in sky coordinates or pixel coordinates of the model frame.
        **kwargs: dict, optional
            Additional arguments passed on to the base component (for subclasses with a different
            base component, e.g. :py:class:`~scarlet2.MovingSource`).

        Examples
        --------
        A source declaration is restricted to a context of a :py:class:`~scarlet2.Scene`,
        which defines the :py:class:`~scarlet2.Frame` of the entire model.

        >>> with Scene(model_frame) as scene:
        >>>    source = Source(center, spectrum, morphology)

        A source can comprise one or multiple :py:class:`~scarlet2.Component`,
        which can be added by :py:func:`add_component` or operators `+=`
        (for an additive component) or `*=` (for a multiplicative component).

        >>> with Scene(model_frame) as scene:
        >>>    source = Source(center, spectrum, morphology)
        >>>    source *= DustComponent(center, dust_spectrum, dust_morphology)
        """
        # set the base component
        super().__init__(center, spectrum, morphology, bbox=bbox, **kwargs)
        # create the empty component list
        self.components = list()
        self.component_ops = list()

        # add this source to the active scene
        try:
            sources = Scenery.scene.sources
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise

        sources.append(self)

    def add_component(self, component, op):
        """Add `component` to this source

        Parameters
        ----------
        component: :py:class:`~scarlet2.Component`
            The component to include in this source. It will be combined with the
            previous component according to the operator `op`.
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
        """What to run when Source is called"""
        base = super()
        model = base.__call__()
        for component, op in zip(self.components, self.component_ops, strict=False):
            model_ = component()
            # cut out regions from model and model_
            bbox, bbox_ = overlap_slices(self.bbox, component.bbox, return_boxes=True)
            sub_model = jax.lax.dynamic_slice(model, bbox.start, bbox.shape)
            sub_model_ = jax.lax.dynamic_slice(model_, bbox_.start, bbox_.shape)
            # combine with operator
            sub_model = op(sub_model, sub_model_)
            # add model_ back in full model
            model = jax.lax.dynamic_update_slice(model, sub_model, bbox.start)
        return model


class PointSource(Source):
    """Point source model"""

    def __init__(self, center, spectrum):
        """Model for point sources

        Because the morphology is determined by the model PSF, it does not need to be provided.

        Parameters
        ----------
        center: array, :py:class:`astropy.coordinates.SkyCoord`
            Center position. If given as astropy sky coordinate, it will be
            transformed with the WCS of the model frame.
        spectrum: array, :py:class:`~scarlet2.Spectrum`
            The spectrum of the point source.

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


class MovingSource(Source, MovingComponent):
    """Source that moves between epochs

    The base model is a :py:class:`~scarlet2.MovingComponent`: the same morphology
    in every channel, a free spectrum amplitude in every channel, and a different
    center in every epoch. Additional components can be added as for any
    :py:class:`~scarlet2.Source`.
    """

    def __init__(self, center, spectrum, morphology, epochs=None):
        """
        Parameters
        ----------
        center: array, :py:class:`astropy.coordinates.SkyCoord`
            Center positions, one for every channel (if `epochs` is `None`) or one for every
            epoch in `epochs`, with shape `(N, 2)` in pixel coordinates of the model frame,
            or as an array of astropy sky coordinates.
        spectrum: array, :py:class:`~scarlet2.Spectrum`
            The spectrum of the source, with one element for every channel of the model frame.
        morphology: array, :py:class:`~scarlet2.Morphology`
            The morphology of the source, the same in all epochs.
        epochs: list, optional
            Identifiers of the epochs, in the order of `center`. Every epoch in the `EPOCH`
            component of the model frame channels needs to be listed. If `None`, every
            channel has its own center.

        Examples
        --------
        >>> with Scene(model_frame) as scene:
        >>>    source = MovingSource(centers, spectrum, morphology)
        """
        super().__init__(center, spectrum, morphology, epochs=epochs)


class MovingPointSource(MovingSource):
    """Point source that moves between epochs"""

    def __init__(self, center, spectrum, epochs=None):
        """Model for moving point sources

        Because the morphology is determined by the model PSF, it does not need to be provided.

        Parameters
        ----------
        center: array, :py:class:`astropy.coordinates.SkyCoord`
            Center positions, one for every channel (if `epochs` is `None`) or one for every
            epoch in `epochs`, with shape `(N, 2)` in pixel coordinates of the model frame,
            or as an array of astropy sky coordinates.
        spectrum: array, :py:class:`~scarlet2.Spectrum`
            The spectrum of the point source, with one element for every channel of the model frame.
        epochs: list, optional
            Identifiers of the epochs, in the order of `center`. Every epoch in the `EPOCH`
            component of the model frame channels needs to be listed. If `None`, every
            channel has its own center.
        """
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Source can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise
        if frame.psf is None:
            raise AttributeError("MovingPointSource can only be create with a PSF in the model frame")
        morphology = frame.psf.morphology

        super().__init__(center, spectrum, morphology, epochs=epochs)


class SourceValidator(metaclass=ValidationMethodCollector):
    """A class containing all of the validation checks for Source objects.

    Note that the metaclass is defined as `MethodCollector`, which collects all
    validation methods in this class into a single class attribute list called
    `validation_checks`. This allows for easy iteration over all checks."""

    def __init__(self, source: Source):
        self.source = source
