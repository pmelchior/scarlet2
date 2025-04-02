"""PSF-related classes"""
import jax.numpy as jnp

from .module import Module
from .morphology import GaussianMorphology


class PSF(Module):
    """PSF base class"""
    pass


class ArrayPSF(PSF):
    """PSF defined by an image array

    Warnings
    --------
    The number of pixels in `morphology` should be odd, and the centroid of the PSF image should be in the central pixel.
    If that is not the case, one creates an effective shift by the PSF, which is not captured by the coordinate
    convention of the frame, e.g. its :py:attr:`~scarlet2.Frame.wcs`.

    See :issue:`96` from more details.
    """
    morphology: jnp.ndarray
    """The PSF morphology image. Can be 2D (height, width) or 3D (channel, height, width)"""

    def __call__(self):
        """Evaluate PSF

        Returns
        -------
        array
            2D image, normalized to total flux=1
        """
        return self.morphology / self.morphology.sum((-2, -1), keepdims=True)


class GaussianPSF(PSF):
    """Gaussian-shaped PSF"""
    morphology: GaussianMorphology
    """Morphology model"""

    def __init__(self, sigma):
        """Initialize Gaussian PSF

        Parameters
        ----------
        sigma: float
            Standard deviation of Gaussian
        """
        self.morphology = GaussianMorphology(sigma)

    def __call__(self):
        morph = self.morphology()
        morph /= morph.sum((-2,-1), keepdims=True)
        return morph
