import equinox as eqx
import jax.numpy as jnp
import jax.scipy

from .bbox import Box
from .module import Module, Parameter


class Morphology(Module):
    bbox: Box = eqx.field(static=True, init=False)

    def normalize(self, x):
        return x / x.max()

    def center_bbox(self, center):
        if isinstance(center, Parameter):
            center_ = center.value
        else:
            center_ = center
        center_ = tuple(_.item() for _ in center_.astype(int))
        self.bbox.set_center(center_)


class ArrayMorphology(Morphology):
    data: jnp.array

    def __init__(self, data):
        self.data = data
        super().__post_init__()
        self.bbox = Box(self.data.shape)

    def __call__(self):
        return self.normalize(self.data)


class ProfileMorphology(Morphology):
    center: jnp.array
    size: float
    ellipticity: (None, jnp.array)

    def __init__(self, center, size, ellipticity=None, bbox=None):

        # define radial profile function
        self.center = center
        self.size = size
        if ellipticity is None:
            self.ellipticity = jnp.zeros((2,))
        else:
            self.ellipticity = ellipticity

        super().__post_init__()

        if bbox is None:
            max_size = jnp.max(self.size)
            # explicit call to int() to avoid bbox sizes being jax-traced
            size = 10 * int(jnp.ceil(max_size))
            if size % 2 == 0:
                size += 1
            center_int = jnp.floor(self.center)
            shape = (size, size)
            origin = (int(center_int[0]) - size // 2, int(center_int[1]) - size // 2)
            bbox = Box(shape, origin=origin)
        self.bbox = bbox

    @property
    def g(self):
        g_factor = 1 / (1. + jnp.sqrt(1. - (self.ellipticity[0]**2 + self.ellipticity[1]**2)))
        return self.ellipticity * g_factor

    def f(self, R2):
        raise NotImplementedError

    def __call__(self):

        _Y = jnp.arange(self.bbox.shape[-2], dtype=float) + self.bbox.origin[-2] - self.center[-2]
        _X = jnp.arange(self.bbox.shape[-1], dtype=float) + self.bbox.origin[-1] - self.center[-1]

        g1, g2 = self.g

        __X = ((1 - g1) * _X[None, :] - g2 * _Y[:, None]) / jnp.sqrt(
            1 - (g1 ** 2 + g2 ** 2)
        )
        __Y = (-g2 * _X[None, :] + (1 + g1) * _Y[:, None]) / jnp.sqrt(
            1 - (g1 ** 2 + g2 ** 2)
        )
        R2 = __Y ** 2 + __X ** 2

        R2 /= self.size ** 2
        R2 = jnp.maximum(R2, 1e-3)  # prevents infs at R2 = 0
        morph = self.normalize(self.f(R2))
        return morph


class GaussianMorphology(ProfileMorphology):

    def __init__(self, center, size, ellipticity=None, bbox=None):
        super().__init__(center, size, ellipticity=ellipticity, bbox=bbox)

    def f(self, R2):
        return jnp.exp(-R2 / 2)

    def __call__(self):

        # faster circular 2D Gaussian: instead of N^2 evaluations, use outer product of 2 1D Gaussian evals
        if self.ellipticity==jnp.zeros((2,)):

            _Y = jnp.arange(self.bbox.shape[-2]) + self.bbox.origin[-2] - self.center[-2]
            _X = jnp.arange(self.bbox.shape[-1]) + self.bbox.origin[-1] - self.center[-1]

            # with pixel integration
            f = lambda x, s: 0.5 * (
                    1 - jax.scipy.special.erfc((0.5 - x) / jnp.sqrt(2) / s) +
                    1 - jax.scipy.special.erfc((0.5 + x) / jnp.sqrt(2) / s)
            )
            # # without pixel integration
            # f = lambda x, s: jnp.exp(-(x ** 2) / (2 * s ** 2)) / (jnp.sqrt(2 * jnp.pi) * s)

            return self.normalize(jnp.outer(f(_Y, self.size), f(_X, self.size)))

        else:
            return super().__call__()


class SersicMorphology(ProfileMorphology):
    n: float

    def __init__(self, n, center, size, ellipticity=None, bbox=None):
        self.n = n
        super().__init__(center, size, ellipticity=ellipticity, bbox=bbox)

    def f(self, R2):
        n = self.n
        n2 = n * n
        # simplest form of bn: Capaccioli (1989)
        # bn = 1.9992 * n - 0.3271
        #
        # better treatment in  Ciotti & Bertin (1999), eq. 18
        # stable to n > 0.36, with errors < 10^5
        bn = 2 * n - 0.333333 + 0.009877 / n + 0.001803 / n2 + 0.000114 / (n2 * n) - 0.000072 / (n2 * n2)

        # MacArthur, Courteau, & Holtzman (2003), eq. A2
        # much more stable for n < 0.36
        # not using it here to avoid if clause in jitted code
        #     bn = 0.01945 - 0.8902 * n + 10.95 * n2 - 19.67 * n2 * n + 13.43 * n2 * n2

        # Graham & Driver 2005, eq. 1
        # we're given R^2, so we use R2^(0.5/n) instead of 1/n
        return jnp.exp(-bn * (R2 ** (0.5 / n) - 1))
