import equinox as eqx
import jax.numpy as jnp
import jax.scipy

from .bbox import Box
from .module import Module


class Morphology(Module):
    bbox: Box = eqx.field(static=True, init=False)
    center: (jnp.ndarray, None) = None

    def set_center(self, center):
        self.set('center', center)
        center_ = tuple(_.item() for _ in self.center.astype(int))
        self.bbox.set_center(center_)


class ArrayMorphology(Morphology):
    data: jnp.array

    def __init__(self, data):
        self.data = data
        super().__post_init__()
        self.bbox = Box(self.data.shape)

    def __call__(self):
        return self.data


class GaussianMorphology(Morphology):
    center: jnp.ndarray
    sigma: float

    def __init__(self, center, sigma, bbox=None):

        self.sigma = sigma
        self.center = center

        if bbox is None:
            max_sigma = jnp.max(sigma)
            # explicit call to int() to avoid bbox sizes being jax-traced
            size = 10 * int(jnp.ceil(max_sigma))
            if size % 2 == 0:
                size += 1
            center_int = jnp.floor(self.center)
            shape = (size, size)
            origin = (int(center_int[0]) - size // 2, int(center_int[1]) - size // 2)
            bbox = Box(shape, origin=origin)
        self.bbox = bbox

        super().__post_init__()

    def __call__(self):
        # grid positions in X/Y
        _Y = jnp.arange(self.bbox.shape[-2]) + self.bbox.origin[-2]
        _X = jnp.arange(self.bbox.shape[-1]) + self.bbox.origin[-1]

        # with pixel integration
        f = lambda x, s: 0.5 * (
                1 - jax.scipy.special.erfc((0.5 - x) / jnp.sqrt(2) / s) +
                1 - jax.scipy.special.erfc((0.5 + x) / jnp.sqrt(2) / s)
        )
        # # without pixel integration
        # f = lambda x, s: jnp.exp(-(x ** 2) / (2 * s ** 2)) / (jnp.sqrt(2 * jnp.pi) * s)

        return jnp.outer(f(_Y - self.center[0], self.sigma), f(_X - self.center[1], self.sigma))
