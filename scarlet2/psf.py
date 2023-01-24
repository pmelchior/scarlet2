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
    sigma: (float, jnp.ndarray, Parameter)
    center: (jnp.ndarray, Parameter)
    bbox: Box = eqx.static_field()
    integrate: bool = True

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

        if isinstance(self.sigma, float):
            psf = jnp.outer(f(_Y - self.center[0], self.sigma),
                            f(_X - self.center[1], self.sigma))
        else:
            psf = jnp.stack([jnp.outer(f(_Y - self.center[0], s), f(_X - self.center[1], s)) for s in self.sigma],
                            axis=0)
        return psf
