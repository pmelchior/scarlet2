import equinox as eqx
import jax.numpy as jnp
import jax.scipy

from .bbox import Box
from .module import Module


class Morphology(Module):
    bbox: Box = eqx.field(static=True, init=False)

    def center_bbox(self, center):
        center_ = tuple(_.item() for _ in center.astype(int))
        self.bbox.set_center(center_)


class ArrayMorphology(Morphology):
    data: jnp.array

    def __init__(self, data):
        self.data = data
        super().__post_init__()
        self.bbox = Box(self.data.shape)

    def __call__(self):
        return self.data


class ProfileMorphology(Morphology):
    f: callable
    center: jnp.array
    size: float
    ellipticity: (None, jnp.array)

    def __init__(self, f, center, size, ellipticity=None, bbox=None):

        # define radial profile function
        self.f = f
        self.center = center
        self.size = size
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

    def __call__(self):

        _Y = jnp.arange(self.bbox.shape[-2], dtype=jnp.float) + self.bbox.origin[-2] - self.center[-2]
        _X = jnp.arange(self.bbox.shape[-1], dtype=jnp.float) + self.bbox.origin[-1] - self.center[-1]

        if self.ellipticity is not None:
            R2 = _Y[:, None] ** 2 + _X[None, :] ** 2
        else:
            e1, e2 = self.ellipticity
            __X = ((1 - e1) * _X[None, :] - e2 * _Y[:, None]) / np.sqrt(
                1 - (e1 ** 2 + e2 ** 2)
            )
            __Y = (-e2 * _X[None, :] + (1 + e1) * _Y[:, None]) / np.sqrt(
                1 - (e1 ** 2 + e2 ** 2)
            )
            R2 = __Y ** 2 + __X ** 2

        R2 /= self.size ** 2

        morph = self.f(R2)
        return morph


class GaussianMorphology(ProfileMorphology):

    def __init__(self, center, size, ellipticity=None, bbox=None):
        super().__init__(lambda R2: jnp.exp(-R2 / 2), center, size, ellipticity=ellipticity, bbox=bbox)

    def __call__(self):

        # faster circular 2D Gaussian: instead of N^2 evaluations, use outer product of 2 1D Gaussian evals
        if self.ellipticity is None:

            _Y = jnp.arange(self.bbox.shape[-2]) + self.bbox.origin[-2] - self.center[-2]
            _X = jnp.arange(self.bbox.shape[-1]) + self.bbox.origin[-1] - self.center[-1]

            # with pixel integration
            f = lambda x, s: 0.5 * (
                    1 - jax.scipy.special.erfc((0.5 - x) / jnp.sqrt(2) / s) +
                    1 - jax.scipy.special.erfc((0.5 + x) / jnp.sqrt(2) / s)
            )
            # # without pixel integration
            # f = lambda x, s: jnp.exp(-(x ** 2) / (2 * s ** 2)) / (jnp.sqrt(2 * jnp.pi) * s)

            return jnp.outer(f(_Y, self.size), f(_X, self.size))

        else:
            return super().__call__(self)


class SersicMorphology(ProfileMorphology):
    n: float

    def __init__(self, n, center, size, ellipticity=None, bbox=None):
        self.n = n
        super().__init__(self._f, center, size, ellipticity=ellipticity, bbox=bbox)

    def _f(self, R2):
        n = self.n
        n2 = n * n
        # simplest form of bn: Capaccioli (1989)
        # bn = 1.9992 * n - 0.3271
        #
        # better treatment in  Ciotti & Bertin (1999), eq. 18
        # stable to n > 0.36, with errors < 10^5
        if n > 0.36:
            bn = 2 * n - 0.333333 + 0.009877 / n + 0.001803 / n2 + 0.000114 / (n2 * n) - 0.000072 / (n2 * n2)

        # MacArthur, Courteau, & Holtzman (2003), eq. A2
        # much more stable for n < 0.36
        if n <= 0.36:
            bn = 0.01945 - 0.8902 * n + 10.95 * n2 - 19.67 * n2 * n + 13.43 * n2 * n2

        # Graham & Driver 2005, eq. 1
        # we're given R^2, so we use R2^(0.5/n) instead of 1/n
        return jnp.exp(-bn * (R2 ** (0.5 / self.n) - 1))
