import astropy.units as u
import equinox as eqx
import jax.numpy as jnp
import jax.scipy

from . import Scenery
from . import measure
from .module import Module
from .wavelets import starlet_transform, starlet_reconstruction


class Morphology(Module):

    @property
    def shape(self):
        raise NotImplementedError

    def normalize(self, x):
        return x / x.max()


class ArrayMorphology(Morphology):
    data: jnp.array

    def __init__(self, data):
        self.data = data

    def __call__(self, **kwargs):
        return self.normalize(self.data)

    @property
    def shape(self):
        return self.data.shape


class ProfileMorphology(Morphology):
    size: float
    ellipticity: (None, jnp.array)
    _shape: tuple = eqx.field(static=True, init=False, repr=False)

    def __init__(self, size, ellipticity=None, shape=None):

        if isinstance(size, u.Quantity):
            try:
                size = Scenery.scene.frame.u_to_pixel(size)
            except AttributeError:
                print("`size` defined in astropy units can only be used within the context of a Scene")
                print("Use 'with Scene(frame) as scene: (...)'")
                raise

        self.size = size
        self.ellipticity = ellipticity

        # default shape: square 10x size
        if shape is None:
            # explicit call to int() to avoid bbox sizes being jax-traced
            size = int(jnp.ceil(10 * self.size))
            # odd shapes for unique center pixel
            if size % 2 == 0:
                size += 1
            shape = (size, size)
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def f(self, R2):
        raise NotImplementedError

    def __call__(self, delta_center=jnp.zeros(2)):

        _Y = jnp.arange(-(self.shape[-2] // 2), self.shape[-2] // 2 + 1, dtype=float) + delta_center[-2]
        _X = jnp.arange(-(self.shape[-1] // 2), self.shape[-1] // 2 + 1, dtype=float) + delta_center[-1]

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

    def __call__(self, delta_center=jnp.zeros(2)):

        # faster circular 2D Gaussian: instead of N^2 evaluations, use outer product of 2 1D Gaussian evals
        if self.ellipticity is None:
            _Y = jnp.arange(-(self.shape[-2] // 2), self.shape[-2] // 2 + 1, dtype=float) + delta_center[-2]
            _X = jnp.arange(-(self.shape[-1] // 2), self.shape[-1] // 2 + 1, dtype=float) + delta_center[-1]

            # with pixel integration
            f = lambda x, s: 0.5 * (
                    1 - jax.scipy.special.erfc((0.5 - x) / jnp.sqrt(2) / s) +
                    1 - jax.scipy.special.erfc((0.5 + x) / jnp.sqrt(2) / s)
            )
            # # without pixel integration
            # f = lambda x, s: jnp.exp(-(x ** 2) / (2 * s ** 2)) / (jnp.sqrt(2 * jnp.pi) * s)

            return self.normalize(jnp.outer(f(_Y, self.size), f(_X, self.size)))

        else:
            return super().__call__(delta_center)

    @staticmethod
    def from_image(image):
        assert image.ndim == 2
        center = measure.centroid(image)
        # compute moments and create Gaussian from it
        g = measure.moments(image, center=center, N=2)
        return GaussianMorphology.from_moments(g)

    @staticmethod
    def from_moments(g):
        T = g.size
        ellipticity = g.ellipticity

        # create image of Gaussian with these 2nd moments
        if jnp.isfinite(T) and jnp.isfinite(ellipticity).all():
            morph = GaussianMorphology(T, ellipticity)
        else:
            raise ValueError(
                f"Gaussian morphology not possible with size={T}, and ellipticity={ellipticity}!")
        return morph


class SersicMorphology(ProfileMorphology):
    n: float

    def __init__(self, n, size, ellipticity=None):
        self.n = n
        super().__init__(size, ellipticity=ellipticity)

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
    l1_thresh: float = eqx.field(default=1e-2, static=True)
    positive: bool = eqx.field(default=True, static=True)

    def __call__(self, **kwargs):
        if self.positive:
            f = prox_soft_plus
        else:
            f = prox_soft
        return self.normalize(starlet_reconstruction(f(self.coeffs, self.l1_thresh)))

    @property
    def shape(self):
        return self.coeffs.shape[-2:]  # wavelet coeffs: scales x n1 x n2

    @staticmethod
    def from_image(image, **kwargs):
        # Starlet transform of image (n1,n2) into coefficient with 3 dimensions: (scales+1,n1,n2)
        coeffs = starlet_transform(image)
        return StarletMorphology(coeffs, **kwargs)
