import equinox as eqx
import jax
import jax.numpy as jnp

from .bbox import overlap_slices
from .frame import Frame
from .module import Module


class Scenery:
    # static store for context manager
    scene = None

class Scene(Module):
    frame: Frame = eqx.static_field()
    sources: list

    def __init__(self, frame):
        self.frame = frame
        # TODO: scene does not report sources as parameters because pytrees treat lists as nodes, not leaves
        self.sources = list()

    def __call__(self):
        model = jnp.zeros(self.frame.bbox.shape)

        # TODO: below does not work because one cannot access a list (self.sources) in jitted functions
        # def insert_model(k, model):
        #     source = self.sources[k]
        #     model_ = source() # only model inside its bbox
        #
        #     # cut out region from model, add single source model
        #     bbox, bbox_ = overlap_slices(self.frame.bbox, source.bbox, return_boxes=True)
        #     sub_model = jax.lax.dynamic_slice(model, bbox.start, bbox.shape)
        #     sub_model_ = jax.lax.dynamic_slice(model_, bbox_.start, bbox_.shape)
        #     sub_model += sub_model_
        #
        #     # add model_ back in full model
        #     model = jax.lax.dynamic_update_slice(model, sub_model, bbox.start)
        #     return model
        #
        # model = jax.lax.fori_loop(0, 2, insert_model, model)

        for source in self.sources:
            model_ = source()

            # cut out region from model, add single source model
            bbox, bbox_ = overlap_slices(self.frame.bbox, source.bbox, return_boxes=True)
            sub_model = jax.lax.dynamic_slice(model, bbox.start, bbox.shape)
            sub_model_ = jax.lax.dynamic_slice(model_, bbox_.start, bbox_.shape)
            sub_model += sub_model_

            # add model_ back in full model
            model = jax.lax.dynamic_update_slice(model, sub_model, bbox.start)
        return model

    def __enter__(self):
        # context manager to register sources
        # purpose is to provide scene.frame to source inits that will need some of its information
        # also allows us to append the sources automatically to the scene
        Scenery.scene = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Scenery.scene = None
    
