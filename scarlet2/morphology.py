import equinox as eqx
import jax.numpy as jnp
import jax.scipy

from .bbox import Box
from .module import Module
from .wavelets import starlet_transform, starlet_reconstruction


class Morphology(Module):
    bbox: Box = eqx.field(static=True, init=False)

    def normalize(self, x):
        return x / x.max()

    def center_bbox(self, center):
        self.bbox.set_center(center.astype(int))


class ArrayMorphology(Morphology):
    data: jnp.array

    def __init__(self, data):
        self.data = data
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
        self.ellipticity = ellipticity

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

    def f(self, R2):
        raise NotImplementedError

    def __call__(self):

        _Y = jnp.arange(self.bbox.shape[-2], dtype=float) + self.bbox.origin[-2] - self.center[-2]
        _X = jnp.arange(self.bbox.shape[-1], dtype=float) + self.bbox.origin[-1] - self.center[-1]

        if self.ellipticity is None:
            R2 = _Y[:, None] ** 2 + _X[None, :] ** 2
        else:
            e1, e2 = self.ellipticity
            g_factor = 1 / (1. + jnp.sqrt(1. - (e1 ** 2 + e2 ** 2)))
            g1, g2 = self.ellipticity * g_factor
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

    def f(self, R2):
        return jnp.exp(-R2 / 2)

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


prox_plus = lambda x: jnp.maximum(x, 0)
prox_soft = lambda x, thresh: jnp.sign(x) * prox_plus(jnp.abs(x) - thresh)
prox_soft_plus = lambda x, thresh: prox_plus(prox_soft(x, thresh))


class StarletMorphology(Morphology):
    coeffs: jnp.ndarray
    l1_thresh: float = eqx.field(static=True)
    positive: bool = eqx.field(static=True)

    def __init__(self, coeffs, l1_thresh=1e-2, positive=True, bbox=None):
        if bbox is None:
            # wavelet coeffs: scales x n1 x n2
            bbox = Box(coeffs.shape[-2:])
        self.bbox = bbox

        self.coeffs = coeffs
        self.l1_thresh = l1_thresh
        self.positive = positive

    def __call__(self):
        if self.positive:
            f = prox_soft_plus
        else:
            f = prox_soft
        return self.normalize(starlet_reconstruction(f(self.coeffs, self.l1_thresh)))

    @staticmethod
    def from_image(image, **kwargs):
        # Starlet transform of image (n1,n2) into coefficient with 3 dimensions: (scales+1,n1,n2)
        coeffs = starlet_transform(image)
        return StarletMorphology(coeffs, **kwargs)
