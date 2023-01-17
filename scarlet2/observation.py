import equinox as eqx
import jax.numpy as jnp

from .fft import convolve
from .frame import Frame
from .module import Module


class Observation(Module):
    data: jnp.ndarray = eqx.static_field()
    weights: jnp.ndarray = eqx.static_field()
    frame: Frame

    def __init__(self, data, weights, psf=None, wcs=None, channels=None):
        self.data = jnp.asarray(data)
        self.weights = jnp.asarray(weights)
        if channels is None:
            channels = range(data.shape[0])
        self.frame = Frame(data.shape, psf, wcs, channels)

    def __call__(self, model):
        # render the model in the frame of the observation
        # here: only convolution needed
        return convolve(model, self.frame.psf(), axes=(1, 2))

    def log_likelihood(self, model):
        # rendered model
        model_ = self.__call__(model)
        # normalization of the single-pixel likelihood:
        # 1 / [(2pi)^1/2 (sigma^2)^1/2]
        # with inverse variance weights: sigma^2 = 1/weight
        # full likelihood is sum over all (unmasked) pixels in data
        D = jnp.prod(jnp.asarray(self.data.shape)) - jnp.sum(self.weights == 0)
        log_norm = D / 2 * jnp.log(2 * jnp.pi)
        log_like = -jnp.sum(self.weights * (model_ - self.data) ** 2) / 2
        return log_like - log_norm
