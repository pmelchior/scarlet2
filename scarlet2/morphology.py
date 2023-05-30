import equinox as eqx
import jax.numpy as jnp
import jax.scipy

from .bbox import Box
from .module import Parameter, Module


class Morphology(Module):
    bbox: Box = eqx.static_field()
    center: (Parameter, None) = None

    def set_center(self, center):
        object.__setattr__(self, 'center', center)
        center_ = tuple(_.item() for _ in center.value.astype(int))
        self.bbox.set_center(center_)


class ArrayMorphology(Morphology, Parameter):
    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)
        self.bbox = Box(self.value.shape)


class GaussianMorphology(Morphology):
    sigma: Parameter

    def __init__(self, center, sigma, bbox=None):

        if not isinstance(sigma, Parameter):
            sigma = Parameter(sigma, fixed=True)
        self.sigma = sigma
        if not isinstance(center, Parameter):
            center = Parameter(center, fixed=True)
        self.center = center

        if bbox is None:
            sigma = self.sigma()
            max_sigma = jnp.max(sigma)
            # explicit call to int() to avoid bbox sizes being jax-traced
            size = 10 * int(jnp.ceil(max_sigma))
            if size % 2 == 0:
                size += 1
            center_int = jnp.floor(self.center())
            shape = (size, size)
            origin = (int(center_int[0]) - size // 2, int(center_int[1]) - size // 2)
            bbox = Box(shape, origin=origin)
        self.bbox = bbox

    def __call__(self):
        # grid positions in X/Y
        _Y = jnp.arange(self.bbox.shape[-2]) + self.bbox.origin[-2]
        _X = jnp.arange(self.bbox.shape[-1]) + self.bbox.origin[-1]

        center = self.center()
        sigma = self.sigma()

        # with pixel integration
        f = lambda x, s: 0.5 * (
                1 - jax.scipy.special.erfc((0.5 - x) / jnp.sqrt(2) / s) +
                1 - jax.scipy.special.erfc((0.5 + x) / jnp.sqrt(2) / s)
        )
        # # without pixel integration
        # f = lambda x, s: jnp.exp(-(x ** 2) / (2 * s ** 2)) / (jnp.sqrt(2 * jnp.pi) * s)

        return jnp.outer(f(_Y - center[0], sigma), f(_X - center[1], sigma))

    # def __call__(self):
    #     psf = self.__call__() # GaussianPSF.__call__(self)
    #     # morphologies should have peak pixel value =~ 1
    #     # could be removed for performance gains
    #     psf /= psf.max()
    #     return psf
