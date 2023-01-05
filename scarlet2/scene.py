import equinox as eqx
import jax
import jax.numpy as jnp

from .frame import Frame
from .module import Module


class Scene(Module):
    frame: Frame = eqx.static_field()
    sources: list

    def __init__(self, frame, sources):
        self.frame = frame
        self.sources = sources

    def __call__(self):
        model = jnp.zeros(self.frame.bbox.shape)
        # TODO!!!
        # def insert_model(k, model):
        #    model_ = self.sources[k]() # only model inside its bbox
        # model = jax.lax.dynamic_update_slice(model, model_, self.sources[k].bbox.shape)
        # model = jax.lax.fori_loop(0, len(self.sources), insert_model, model)
        for source in self.sources:
            model = jax.lax.dynamic_update_slice(model, source(), source.bbox.shape)
        return model
