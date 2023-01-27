import equinox as eqx
import jax.numpy as jnp
import jax.scipy

from .bbox import Box
from .module import Module, Parameter


class PSF(Module):
    def __call__(self):
        raise NotImplementedError


class ArrayPSF(Parameter, PSF):
    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)


class GaussianPSF(PSF):
    center: Parameter
    sigma: Parameter
    bbox: Box = eqx.static_field()
    integrate: bool = True

    def __init__(self, center, sigma, bbox=None, integrate=True):

        if not isinstance(sigma, Parameter):
            sigma = Parameter(sigma, fixed=True)
        self.sigma = sigma
        if not isinstance(center, Parameter):
            center = Parameter(center, fixed=True)
        self.center = center

        if bbox is None:
            sigma = self.sigma()
            len_sigma = 0 if isinstance(sigma, float) else len(sigma)
            max_sigma = jnp.max(sigma)
            # explicit call to int() to avoid bbox sizes being jax-traced
            size = 10 * int(jnp.ceil(max_sigma))
            if size % 2 == 0:
                size += 1
            center_int = jnp.floor(self.center())
            shape = (size, size)
            origin = (int(center_int[0]) - size // 2, int(center_int[1]) - size // 2)
            if len_sigma:
                shape = (len_sigma, *shape)
                origin = (0, *origin)
            bbox = Box(shape, origin=origin)
        self.bbox = bbox

        self.integrate = integrate

    def __call__(self):
        # grid positions in X/Y
        _Y = jnp.arange(self.bbox.shape[-2]) + self.bbox.origin[-2]
        _X = jnp.arange(self.bbox.shape[-1]) + self.bbox.origin[-1]

        if self.integrate:
            f = lambda x, s: 0.5 * (
                    1 - jax.scipy.special.erfc((0.5 - x) / jnp.sqrt(2) / s) +
                    1 - jax.scipy.special.erfc((0.5 + x) / jnp.sqrt(2) / s)
            )
        else:
            f = lambda x, s: jnp.exp(-(x ** 2) / (2 * s ** 2)) / (jnp.sqrt(2 * jnp.pi) * s)

        center = self.center()
        sigma = self.sigma()
        len_sigma = 0 if isinstance(sigma, float) else len(sigma)
        if len_sigma:
            psf = jnp.stack([jnp.outer(f(_Y - center[0], s), f(_X - center[1], s)) for s in sigma], axis=0)
        else:
            psf = jnp.outer(f(_Y - center[0], sigma), f(_X - center[1], sigma))
        return psf
