import astropy.units as u
import equinox as eqx
import jax.numpy as jnp
import jax.scipy

from . import Scenery, measure
from .module import Module
from .wavelets import starlet_reconstruction, starlet_transform


class Morphology(Module):
    """Morphology base class"""

    @property
    def shape(self):
        """Shape (2D) of the morphology model"""
        raise NotImplementedError


class ProfileMorphology(Morphology):
    """Base class for morphologies based on a radial profile"""

    size: float
    """Size of the profile

    Can be given as an astropy angle, which will be transformed with the WCS of the current
    :py:class:`~scarlet2.Scene`.
    """
    ellipticity: (None, jnp.array)
    """Ellipticity of the profile"""
    _shape: tuple = eqx.field(init=False, repr=False)

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
        """Shape of the bounding box for the profile.

        If not set during `__init__`,  uses a square box with an odd number of pixels
        not smaller than `10*size`.
        """
        return self._shape

    def f(self, r2):
        """Radial profile function

        Parameters
        ----------
        r2: float or array
            Radius (distance from the center) squared
        """
        raise NotImplementedError

    def __call__(self, delta_center=jnp.zeros(2)):  # noqa: B008
        """Evaluate the model"""
        _y = jnp.arange(-(self.shape[-2] // 2), self.shape[-2] // 2 + 1, dtype=float) + delta_center[-2]
        _x = jnp.arange(-(self.shape[-1] // 2), self.shape[-1] // 2 + 1, dtype=float) + delta_center[-1]

        if self.ellipticity is None:
            r2 = _y[:, None] ** 2 + _x[None, :] ** 2
        else:
            e1, e2 = self.ellipticity
            g_factor = 1 / (1.0 + jnp.sqrt(1.0 - (e1**2 + e2**2)))
            g1, g2 = self.ellipticity * g_factor
            __x = ((1 - g1) * _x[None, :] - g2 * _y[:, None]) / jnp.sqrt(1 - (g1**2 + g2**2))
            __y = (-g2 * _x[None, :] + (1 + g1) * _y[:, None]) / jnp.sqrt(1 - (g1**2 + g2**2))
            r2 = __y**2 + __x**2

        r2 /= self.size**2
        r2 = jnp.maximum(r2, 1e-3)  # prevents infs at R2 = 0
        morph = self.f(r2)
        return morph


class GaussianMorphology(ProfileMorphology):
    """Gaussian radial profile"""

    def f(self, r2):
        """Radial profile function

        Parameters
        ----------
        r2: float or array
            Radius (distance from the center) squared
        """
        return jnp.exp(-r2 / 2)

    def __call__(self, delta_center=jnp.zeros(2)):  # noqa: B008
        """Evaluate the model"""
        # faster circular 2D Gaussian: instead of N^2 evaluations, use outer product of 2 1D Gaussian evals
        if self.ellipticity is None:
            _y = jnp.arange(-(self.shape[-2] // 2), self.shape[-2] // 2 + 1, dtype=float) + delta_center[-2]
            _x = jnp.arange(-(self.shape[-1] // 2), self.shape[-1] // 2 + 1, dtype=float) + delta_center[-1]

            # with pixel integration
            f = lambda x, s: 0.5 * (  # noqa: E731
                1
                - jax.scipy.special.erfc((0.5 - x) / jnp.sqrt(2) / s)
                + 1
                - jax.scipy.special.erfc((0.5 + x) / jnp.sqrt(2) / s)
            )
            # # without pixel integration
            # f = lambda x, s: jnp.exp(-(x ** 2) / (2 * s ** 2)) / (jnp.sqrt(2 * jnp.pi) * s)

            return jnp.outer(f(_y, self.size), f(_x, self.size))

        else:
            return super().__call__(delta_center)

    @staticmethod
    def from_image(image):
        """Create Gaussian radial profile from the 2nd moments of `image`

        Parameters
        ----------
        image: array
            2D array to measure :py:class:`~scarlet2.measure.Moments` from.

        Returns
        -------
        GaussianMorphology
        """
        assert image.ndim == 2
        center = measure.centroid(image)
        # compute moments and create Gaussian from it
        g = measure.moments(image, center=center, N=2)
        return GaussianMorphology.from_moments(g, shape=image.shape)

    @staticmethod
    def from_moments(g, shape=None):
        """Create Gaussian radial profile from the moments `g`

        Parameters
        ----------
        g: :py:class:`~scarlet2.measure.Moments`
            Moments, order >= 2
        shape: tuple
            Shape of the bounding box

        Returns
        -------
        GaussianMorphology
        """
        t = g.size
        ellipticity = g.ellipticity

        # create image of Gaussian with these 2nd moments
        if jnp.isfinite(t) and jnp.isfinite(ellipticity).all():
            morph = GaussianMorphology(t, ellipticity, shape=shape)
        else:
            raise ValueError(
                f"Gaussian morphology not possible with size={t}, and ellipticity={ellipticity}!"
            )
        return morph


class SersicMorphology(ProfileMorphology):
    """Sersic radial profile"""

    n: float
    """Sersic index"""

    def __init__(self, n, size, ellipticity=None, shape=None):
        self.n = n
        super().__init__(size, ellipticity=ellipticity, shape=shape)

    def f(self, r2):
        """Radial profile function

        Parameters
        ----------
        r2: float or array
            Radius (distance from the center) squared
        """
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
        return jnp.exp(-bn * (r2 ** (0.5 / n) - 1))


prox_plus = lambda x: jnp.maximum(x, 0)  # noqa: E731
prox_soft = lambda x, thresh: jnp.sign(x) * prox_plus(jnp.abs(x) - thresh)  # noqa: E731
prox_soft_plus = lambda x, thresh: prox_plus(prox_soft(x, thresh))  # noqa: E731


class StarletMorphology(Morphology):
    """Morphology in the starlet basis

    See Also
    --------
    scarlet2.wavelets.Starlet
    """

    coeffs: jnp.ndarray
    """Starlet coefficients"""
    l1_thresh: float = eqx.field(default=0)
    """L1 threshold for coefficient to create sparse representation"""
    positive: bool = eqx.field(default=True)
    """Whether the coefficients are restricted to non-negative values"""

    def __call__(self, **kwargs):
        """Evaluate the model"""
        f = prox_soft_plus if self.positive else prox_soft
        return starlet_reconstruction(f(self.coeffs, self.l1_thresh))

    @property
    def shape(self):
        """Shape (2D) of the morphology model"""
        return self.coeffs.shape[-2:]  # wavelet coeffs: scales x n1 x n2

    @staticmethod
    def from_image(image, **kwargs):
        """Create starlet morphology from `image`

        Parameters
        ----------
        image: array
            2D image array to determine coefficients from.
        kwargs: dict
            Additional arguments for `__init__`

        Returns
        -------
        StarletMorphology
        """
        # Starlet transform of image (n1,n2) into coefficient with 3 dimensions: (scales+1,n1,n2)
        coeffs = starlet_transform(image)
        return StarletMorphology(coeffs, **kwargs)
