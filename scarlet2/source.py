import copy
import operator

import equinox as eqx
import jax
import jax.numpy as jnp

from .bbox import overlap_slices
from .module import Module
from .morphology import Morphology
from .scene import Scenery
from .spectrum import Spectrum


class Component(Module):
    spectrum: Spectrum
    morphology: Morphology

    def __init__(self, center, spectrum, morphology):
        self.spectrum = spectrum
        self.morphology = morphology
        self.morphology.center_bbox(center)
        super().__post_init__()

    def __call__(self):
        # Boxed model
        return self.spectrum()[:, None, None] * self.morphology()[None, :, :]

    @property
    def bbox(self):
        return self.spectrum.bbox @ self.morphology.bbox


class DustComponent(Component):
    def __call__(self):
        return jnp.exp(-super().__call__())


class Source(Component):
    components: list
    component_ops: list = eqx.field(static=True)

    def __init__(self, center, spectrum, morphology):
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
        morphology.set("center", center)
        super().__init__(center, spectrum, morphology)
